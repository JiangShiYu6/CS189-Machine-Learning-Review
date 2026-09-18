"""Reproduce v2 inference sweep, select on development data, then audit once."""
from pathlib import Path
import argparse,gc,json
import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from sft_pipeline import load_base,prepare_data,summarize,save_submission,LETTERS,predict_choices
from improve_inference import score_records

HERE=Path(__file__).resolve().parent
OUT=HERE/'results_v2'


def data():
    revisions=json.loads((HERE/'source_revisions.json').read_text())
    return revisions,prepare_data(HERE/'data',revisions,HERE/'hw5_sample_eval.csv',HERE/'kaggle_test.csv')


def load(name,revisions):
    model,tokenizer=load_base(revisions['model'])
    if name!='base':model=PeftModel.from_pretrained(model,HERE/'models'/name)
    return model,tokenizer


def average(frames):
    result=frames[0].copy()
    assert all(f.id.tolist()==result.id.tolist() for f in frames)
    probabilities=[np.mean([np.asarray(f.iloc[i].choice_probabilities,dtype=float)
                            for f in frames],axis=0) for i in range(len(result))]
    # Audit/development use four options, while course input may vary in length.
    result['choice_probabilities']=[list(p) for p in probabilities]
    result['prediction']=[LETTERS[int(np.argmax(p))] for p in probabilities]
    result['output']=[f'\\boxed{{{p}}}' for p in result.prediction]
    result['correct']=result.prediction==result.answer
    return result


def sweep():
    revisions,(_,dev,_,_,_)=data();OUT.mkdir(exist_ok=True)
    metrics={}
    for name in ['base','final','narrow']:
        model,tok=load(name,revisions)
        for style in ['boxed','direct','answer','raw']:
            for rotations in [1,5]:
                key=f'{name}_{style}_r{rotations}'
                frame=score_records(model,tok,dev,style,rotations)
                frame.to_json(OUT/f'{key}.json',orient='records',indent=2)
                metrics[key]=summarize(frame);print(key,metrics[key],flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    (OUT/'dev_metrics.json').write_text(json.dumps(metrics,indent=2))


def quality(metrics):
    return .65*metrics['specialized']['accuracy']+.35*metrics['general']['accuracy']


def select():
    frames={p.stem:pd.read_json(p) for p in OUT.glob('*_r*.json') if p.stem.endswith(('_r1','_r5'))}
    candidates={name:{'metrics':summarize(frame),'components':[name]} for name,frame in frames.items()}
    new_names=[name for name in candidates if name.startswith('v2_epoch')]
    assert len(new_names)==4,'Both new epochs and rotation modes must be evaluated'
    best_new=max(new_names,key=lambda k:quality(candidates[k]['metrics']))
    groups={
        'ensemble_base_narrow':['base_boxed_r5','narrow_boxed_r5'],
        'ensemble_base_final':['base_boxed_r5','final_boxed_r5'],
        'ensemble_old_three':['base_boxed_r5','final_boxed_r5','narrow_boxed_r5'],
        'ensemble_base_new':['base_boxed_r5',best_new]}
    for name,components in groups.items():
        frame=average([frames[k] for k in components]);frames[name]=frame
        candidates[name]={'metrics':summarize(frame),'components':components}
        frame.to_json(OUT/f'{name}_dev.json',orient='records',indent=2)
    for name,item in candidates.items():
        item['utility']=quality(item['metrics'])
        item['eligible']=(not name.startswith('base_') and item['metrics']['general']['accuracy']>=73/142-.02)
    chosen=max((k for k in candidates if candidates[k]['eligible']),key=lambda k:candidates[k]['utility'])
    result={'selected':chosen,'components':candidates[chosen]['components'],'candidates':candidates,
            'note':'Selected solely on previously observed development data; audit was not consulted.'}
    (OUT/'selection.json').write_text(json.dumps(result,indent=2))
    print('SELECTED',chosen,candidates[chosen],flush=True)


def infer_components(components,records,revisions):
    frames=[]
    for key in components:
        name,style,rotation=key.rsplit('_',2)
        model,tok=load(name,revisions)
        frames.append(score_records(model,tok,records,style,int(rotation[1:])))
        del model;gc.collect();torch.cuda.empty_cache()
    return average(frames) if len(frames)>1 else frames[0]


def audit_and_submit():
    revisions,(_,dev,public,hidden,_)=data()
    audit=[json.loads(x) for x in (HERE/'data/audit_v2.jsonl').read_text(encoding='utf-8').splitlines()]
    selection=json.loads((OUT/'selection.json').read_text())
    selected=selection['components']
    # Save this evaluation once after selection. It is not an optimization target.
    if (OUT/'audit_metrics.json').exists():raise RuntimeError('Audit already evaluated; preserve its one-shot status')
    metrics={}
    for name,components in [('base',['base_boxed_r1']),('previous',['final_boxed_r1']),('selected',selected)]:
        if name in ['base','previous']:
            model,tok=load('base' if name=='base' else 'final',revisions)
            frame=predict_choices(model,tok,audit)
            del model;gc.collect();torch.cuda.empty_cache()
        else:
            frame=infer_components(components,audit,revisions)
        frame.to_json(OUT/f'audit_{name}.json',orient='records',indent=2)
        metrics[name]=summarize(frame)
        print('AUDIT',name,metrics[name],flush=True)
    (OUT/'audit_metrics.json').write_text(json.dumps(metrics,indent=2))
    # Course public labels are evaluated only after freezing the chosen pipeline.
    frame=infer_components(selected,hidden,revisions)
    save_submission(frame,HERE/'kaggle_test.csv',HERE/'submission_v2.csv')
    frame.to_json(OUT/'hidden_predictions.json',orient='records',indent=2)
    public_frame=infer_components(selected,public,revisions)
    public_frame.to_json(OUT/'public_predictions.json',orient='records',indent=2)
    (OUT/'public_metrics.json').write_text(json.dumps(summarize(public_frame),indent=2))
    print('Saved submission_v2.csv',len(frame),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['sweep','select','audit'])
    phase=parser.parse_args().phase
    {'sweep':sweep,'select':select,'audit':audit_and_submit}[phase]()
