"""Lightweight correctness and integrity checks for the improved submission."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from sft_pipeline import fingerprint,read_csv,prepare_data
from improve_inference import positional_reference
from run_improvements import average

HERE=Path(__file__).resolve().parent


def main():
    for text in ['All of the above','A & B','A is correct','None of these','A, B and C']:
        assert positional_reference({'question':'Select one.','choices':[text,'Other']})
    assert not positional_reference({'question':'What is 2+2?','choices':['2','3','4','5']})
    # Mapping/averaging must support records with different numbers of choices.
    a=pd.DataFrame([{'id':'x','answer':'B','choice_probabilities':[.1,.6,.2,.1]},
                    {'id':'y','answer':'E','choice_probabilities':[.1,.1,.1,.1,.6]}])
    merged=average([a,a]);assert merged.prediction.tolist()==['B','E']
    audit_path=HERE/'data/audit_v2.jsonl'
    manifest=json.loads((HERE/'results_v2/audit_manifest.json').read_text())
    audit=[json.loads(x) for x in audit_path.read_text(encoding='utf-8').splitlines()]
    training=[json.loads(x) for x in (HERE/'data/train_v2.jsonl').read_text(encoding='utf-8').splitlines()]
    revisions=json.loads((HERE/'source_revisions.json').read_text())
    _,dev,public,hidden,_=prepare_data(HERE/'data',revisions,HERE/'hw5_sample_eval.csv',HERE/'kaggle_test.csv')
    assert not {fingerprint(r['question']) for r in training}&{fingerprint(r['question']) for r in audit+dev+public+hidden}
    data_manifest=json.loads((HERE/'results_v2/training_data_manifest.json').read_text())
    assert hashlib.sha256(audit_path.read_bytes()).hexdigest()==data_manifest['audit_sha256']
    assert hashlib.sha256((HERE/'data/train_v2.jsonl').read_bytes()).hexdigest()==data_manifest['train_sha256']
    assert any(r['answer']=='E' for r in training)
    selected=json.loads((HERE/'results_v2/selection.json').read_text())
    eligible={k:v for k,v in selected['candidates'].items() if v['eligible']}
    assert selected['selected']==max(eligible,key=lambda k:eligible[k]['utility'])
    for kind in ['base','previous','selected']:
        frame=pd.read_json(HERE/f'results_v2/audit_{kind}.json')
        assert frame.id.tolist()==[r['id'] for r in audit]
        assert (frame.correct==(frame.prediction==frame.answer)).all()
    submission=pd.read_csv(HERE/'submission_v2.csv',dtype=str)
    assert submission.columns.tolist()==['id','prediction']
    assert submission.id.tolist()==read_csv(HERE/'kaggle_test.csv',False).id.astype(str).tolist()
    assert submission.prediction.isin(list('ABCDE')).all()
    predictions=pd.read_json(HERE/'results_v2/hidden_predictions.json')
    assert submission.prediction.tolist()==predictions.prediction.tolist()
    print('PASS: reference protection, mixed-length ensembles, train/dev/audit isolation, data hashes,')
    print('five-choice supervision, selection rule, audit correctness, and 169 submission IDs/labels.')


if __name__=='__main__':main()
