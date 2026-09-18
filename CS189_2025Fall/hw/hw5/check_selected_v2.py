"""Recompute the selected development predictions before the one-shot audit."""
import json
import numpy as np
import pandas as pd
from run_improvements import OUT, data, infer_components


def main():
    selection = json.loads((OUT / 'selection.json').read_text())
    revisions, (_, dev, _, _, _) = data()
    name = selection['selected']
    filename = name + ('_dev' if name.startswith('ensemble_') else '') + '.json'
    expected = pd.read_json(OUT / filename)
    actual = infer_components(selection['components'], dev, revisions)
    actual.to_json(OUT / 'selected_recomputed_dev.json', orient='records', indent=2)
    assert actual.id.tolist() == expected.id.tolist()
    assert actual.prediction.tolist() == expected.prediction.tolist()
    error = max(float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
                for a, b in zip(actual.choice_probabilities, expected.choice_probabilities))
    assert error < 1e-6, error
    result = {'selected': name, 'rows': len(actual), 'predictions_match': True,
              'max_probability_error': error}
    (OUT / 'selected_reload_check.json').write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == '__main__':
    main()
