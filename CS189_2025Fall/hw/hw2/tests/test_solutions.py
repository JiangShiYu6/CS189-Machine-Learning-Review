"""Behavior checks for notebook functions, independent of the Arena download."""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HW = Path(__file__).resolve().parents[1]


def functions():
    namespace = dict(np=np, pd=pd, LogisticRegression=LogisticRegression,
                     results_df=pd.DataFrame(), style_feature_cols=[])
    for filename in ['arena_warmup.ipynb', 'arena_style_control.ipynb']:
        nb = json.loads((HW / filename).read_text(encoding='utf-8'))
        for cell in nb['cells']:
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell['source'])
            if '%' in source or '!' in source:
                # IPython setup cells are not needed for these pure functions.
                if 'def ' not in source:
                    continue
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
            exec(compile(ast.Module(body=definitions, type_ignores=[]), filename, 'exec'), namespace)
    return namespace


def sample():
    return pd.DataFrame([
        dict(question_id='a', model_a='A', model_b='B', winner='model_a'),
        dict(question_id='b', model_a='B', model_b='A', winner='model_b'),
        dict(question_id='c', model_a='A', model_b='B', winner='model_b'),
        dict(question_id='d', model_a='C', model_b='B', winner='tie'),
    ])


def test_selection_requires_both_models_and_excludes_both_tie_types():
    f = functions()
    df = pd.concat([sample(), pd.DataFrame([dict(model_a='A', model_b='B', winner='tie (bothbad)')])])
    selected, decisive = f['subselect_battles'](df, {'A', 'B'})
    assert len(selected) == 4 and len(decisive) == 3


def test_pairwise_counts_both_directions_and_preserves_missing_pairs():
    table = functions()['compute_pairwise_win_fraction'](sample())
    assert np.isclose(table.loc['A', 'B'], 2/3)
    assert np.isclose(table.loc['B', 'A'], 1/3)
    assert np.isnan(table.loc['A', 'C'])
    assert np.isnan(np.diag(table)).all()
    assert table.index.tolist() == table.columns.tolist()


def test_identity_features_are_antisymmetric_and_model_ordered():
    X, y = functions()['turn_into_features'](sample().iloc[:3], ['C', 'B', 'A'])
    np.testing.assert_array_equal(X[0], [0, -1, 1])
    np.testing.assert_array_equal(X[1::2], -X[0::2])
    np.testing.assert_array_equal(y, [1, 0, 0, 1, 0, 1])


def test_style_normalization_sums_nested_counts_and_handles_zero():
    metadata = dict(bold_count_a={'**': 3, '__': 1}, bold_count_b={'**': 1, '__': 1},
                    header_count_a={'h1': 0}, header_count_b={'h1': 0},
                    list_count_a={'ordered': 1, 'unordered': 1}, list_count_b={'ordered': 4},
                    sum_assistant_a_tokens=100, sum_assistant_b_tokens=300)
    df = pd.DataFrame({'conv_metadata': [metadata]})
    result = functions()['add_style_features'](df)
    np.testing.assert_allclose(result.iloc[0, 1:].to_numpy(dtype=float), [1/3, 0, -1/3, -.5])
    assert list(df.columns) == ['conv_metadata']


def test_style_features_reverse_with_labels():
    df = sample().iloc[:3].assign(style_test=[.2, -.4, 0])
    pair = functions()['make_pairwise_feature_df'](df, ['A', 'B'], ['style_test'])
    np.testing.assert_allclose(pair.style_test.to_numpy(), [.2, -.2, -.4, .4, 0, 0])
    assert pair.question_id.tolist() == ['a', 'a', 'b', 'b', 'c', 'c']


def test_confidence_rank_uses_strict_interval_separation():
    f = functions()
    df = pd.DataFrame({'Lower Bound': [2, 1, -1], 'Upper Bound': [3, 2, 0]})
    assert [f['assign_rank'](r, df) for _, r in df.iterrows()] == [1, 1, 3]


def test_bootstrap_reproducible_and_does_not_modify_global_rng():
    f = functions()
    X, y = f['turn_into_features'](sample().iloc[:3], ['A', 'B'])
    np.random.seed(123)
    before = np.random.get_state()
    a, mean, interval = f['get_bootstrapped_score'](X, y, ['A', 'B'], n_bootstrap=12)
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    b, _, _ = f['get_bootstrapped_score'](X, y, ['A', 'B'], n_bootstrap=12)
    pd.testing.assert_frame_equal(a, b)
    assert interval.shape == (2, 2) and np.all(interval[0] <= interval[1])
    assert mean[0] > mean[1]


def test_style_coefficients_never_enter_model_ranking():
    f = functions()
    df = pd.concat([sample().iloc[:3]] * 5, ignore_index=True).assign(style_test=.4)
    result = f['get_sc_category_results'](df, ['A', 'B'], style_features=['style_test'], n_bootstrap=8)
    assert result.loc[result.Model.eq('style_test'), 'Rank'].item() == -1
    assert result.loc[result.Model.isin(['A', 'B']), 'Rank'].between(1, 2).all()
