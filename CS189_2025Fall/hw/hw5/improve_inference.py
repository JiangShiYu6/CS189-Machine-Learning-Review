"""Development experiments for HW5; course CSV answers are never used to tune."""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import torch
from sft_pipeline import LETTERS, build_prompt


def positional_reference(record):
    """Permutation can change the meaning of 'A and B' / 'all above' options."""
    text=record['question']+' '+' '.join(record['choices'])
    return bool(re.search(
        r'\b(?:above|below|both|neither|former|latter|option[s]?|choice[s]?)\b'
        r'|\b(?:all|none)\s+of\b'
        r'|\b[A-E]\s*(?:and|or|only|is|are|&|,)\s*\b',text,re.I))


def prompt_for(record,tokenizer,style):
    options='\n'.join(f'{LETTERS[i]}. {c}' for i,c in enumerate(record['choices']))
    if style=='boxed':
        message=build_prompt(record);prefix='\\boxed{';letters=list(LETTERS)
    elif style=='direct':
        message=(record['question']+'\n\n'+options+'\n\nAnswer with only the letter of the correct option.')
        prefix='';letters=list(LETTERS)
    elif style=='answer':
        message='Choose the correct answer.\n\n'+record['question']+'\n\n'+options
        prefix='The correct answer is';letters=[' '+x for x in LETTERS]
    elif style=='raw':
        return record['question']+'\n'+options+'\nAnswer:', [' '+x for x in LETTERS]
    else:raise ValueError(style)
    return tokenizer.apply_chat_template([{'role':'user','content':message}],tokenize=False,
                                         add_generation_prompt=True)+prefix,letters


def score_records(model,tokenizer,records,style='boxed',rotations=1,batch_size=4):
    """Average mapped option probabilities; never permute position-dependent options."""
    jobs=[]
    for index,record in enumerate(records):
        n=len(record['choices']); repeats=1 if positional_reference(record) else min(rotations,n)
        for shift in range(repeats):
            order=list(range(n))[shift:]+list(range(n))[:shift]
            permuted=dict(record,choices=[record['choices'][i] for i in order])
            prompt,letters=prompt_for(permuted,tokenizer,style)
            ids=[tokenizer.encode(x,add_special_tokens=False) for x in letters]
            if not all(len(x)==1 for x in ids):raise ValueError('Answer token is not atomic')
            jobs.append((index,order,prompt,[x[0] for x in ids]))
    sums=[np.zeros(len(r['choices']),dtype=np.float64) for r in records];counts=np.zeros(len(records))
    tokenizer.padding_side='left';model.eval();device=next(model.parameters()).device
    for start in range(0,len(jobs),batch_size):
        batch=jobs[start:start+batch_size]
        inputs=tokenizer([x[2] for x in batch],padding=True,return_tensors='pt').to(device)
        # Match HF generation: real tokens start at position zero regardless of
        # left-padding length. This avoids BF16 RoPE changes from batch padding.
        positions=inputs['attention_mask'].long().cumsum(-1)-1
        positions.masked_fill_(inputs['attention_mask']==0,1)
        inputs['position_ids']=positions
        with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.bfloat16,
                enabled=device.type=='cuda' and next(model.parameters()).dtype==torch.bfloat16):
            logits=model(**inputs,use_cache=False,logits_to_keep=1).logits[:,-1].float()
        for i,(index,order,prompt,ids) in enumerate(batch):
            probs=torch.softmax(logits[i,ids[:len(order)]],dim=-1).cpu().numpy()
            sums[index][order]+=probs;counts[index]+=1
    rows=[]
    for i,r in enumerate(records):
        probs=sums[i]/counts[i];pred=LETTERS[int(probs.argmax())]
        rows.append({**r,'prediction':pred,'output':f'\\boxed{{{pred}}}',
                     'correct':pred==r.get('answer',''),'choice_probabilities':probs.tolist()})
    return pd.DataFrame(rows)
