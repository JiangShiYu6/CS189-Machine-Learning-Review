"""Validate data isolation, saved predictions, notebook/source agreement and artifacts."""
from pathlib import Path
import ast
import json
import tempfile
import sys
import nbformat
import pandas as pd
from sft_pipeline import (prepare_data, fingerprint, read_csv, build_prompt,
                         parse_choice_from_boxed, summarize, save_submission)

HERE=Path(__file__).resolve().parent


def main():
    revisions=json.loads((HERE/'source_revisions.json').read_text())
    train,heldout,public,hidden,manifest=prepare_data(
        HERE/'data',revisions,HERE/'hw5_sample_eval.csv',HERE/'kaggle_test.csv')
    assert not ({fingerprint(r['question']) for r in train} &
                {fingerprint(r['question']) for r in heldout+public+hidden})
    assert build_prompt(dict(public[0],answer='A'))==build_prompt(dict(public[0],answer='E'))
    for text,expected in [(r'\boxed{C}','C'),(r'\boxed{A} then \boxed{d}','D'),
                          ('B.','B'),('Because A appears in this sentence',None),('',None)]:
        assert parse_choice_from_boxed(text)==expected
    metrics=json.loads((HERE/'results/metrics.json').read_text())
    frames={}
    for file,key in [('base','base'),('narrow','narrow'),('final','mixed')]:
        frame=pd.read_json(HERE/f'results/{file}_predictions.json')
        assert frame.id.tolist()==[r['id'] for r in heldout+public]
        assert (frame.correct==(frame.prediction==frame.answer)).all()
        assert summarize(frame)==metrics[key]
        frames[file]=frame.set_index('id')
    examples=json.loads((HERE/'results/examples.json').read_text())
    for category,rows in examples.items():
        assert len(rows)==2
        for r in rows:
            base,final=(frames[name].loc[r['id']] for name in ['base','final'])
            assert base['group']=='general'
            assert r['base_output']==base.output and r['final_output']==final.output
            assert bool(base.correct)==(category=='forgetting')
            assert bool(final.correct)==(category=='gains')
    submission=pd.read_csv(HERE/'submission.csv',dtype=str)
    assert submission.columns.tolist()==['id','prediction']
    with tempfile.TemporaryDirectory() as temp:
        save_submission(submission,HERE/'kaggle_test.csv',Path(temp)/'valid.csv')
        invalid=submission.copy();invalid.loc[0,'prediction']='F'
        try:save_submission(invalid,HERE/'kaggle_test.csv',Path(temp)/'invalid.csv')
        except ValueError:pass
        else:raise AssertionError('Invalid label was accepted')
    nb=nbformat.read(HERE/'finetuning_tutorial.ipynb',as_version=4)
    code=[c for c in nb.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in code)
    assert not any(o.output_type=='error' for c in code for o in c.outputs)
    module=ast.parse((HERE/'sft_pipeline.py').read_text())
    definitions={node.name:ast.dump(node) for node in module.body if isinstance(node,ast.FunctionDef)}
    embedded={node.name:ast.dump(node) for c in code for node in ast.parse(c.source).body
              if isinstance(node,ast.FunctionDef)}
    assert definitions==embedded, 'Notebook helpers differ from support module'
    for name in ['hw5_paper_answers.pdf','finetuning_tutorial.pdf','hw5_writeup.pdf']:
        assert (HERE/name).read_bytes().startswith(b'%PDF')
    if '--reload-model' in sys.argv:
        from peft import PeftModel
        from sft_pipeline import load_base,predict_choices
        model,tokenizer=load_base(revisions['model'])
        model=PeftModel.from_pretrained(model,HERE/'models/final')
        replay=predict_choices(model,tokenizer,hidden[:4])
        assert replay.prediction.tolist()==submission.prediction.iloc[:4].tolist()
        print('PASS: saved adapter reload reproduces the first hidden-test batch')
    print('PASS: training isolation; label-independent prompts; answer parsing; saved metrics;')
    print('real forgetting/gain examples; submission IDs/labels; executed notebook; helper parity; PDFs.')


if __name__=='__main__':main()
