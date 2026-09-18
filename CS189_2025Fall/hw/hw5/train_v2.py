"""Train a label-only LoRA classifier using the fixed causal LM's next-token logits."""
from pathlib import Path
import json
import random
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import Trainer, TrainingArguments, TrainerCallback
from peft import get_peft_model,LoraConfig
from sft_pipeline import load_base,prepare_data,build_prompt,LETTERS,SEED,summarize
from improve_inference import positional_reference,score_records

HERE=Path(__file__).resolve().parent


def label_dataset(records,tokenizer):
    rng=random.Random(SEED);rows=[];skipped=0
    for r in records:
        order=list(range(len(r['choices'])))
        if not positional_reference(r):rng.shuffle(order)
        answer=LETTERS[order.index(LETTERS.index(r['answer']))]
        prompt=tokenizer.apply_chat_template([{'role':'user','content':build_prompt(
            dict(r,choices=[r['choices'][i] for i in order]))}],tokenize=False,add_generation_prompt=True)+'\\boxed{'
        ids=tokenizer.encode(prompt,add_special_tokens=False)
        if len(ids)>1024:skipped+=1;continue
        target=tokenizer.encode(answer,add_special_tokens=False)
        assert len(target)==1
        rows.append({'input_ids':ids,'attention_mask':[1]*len(ids),'target':target[0]})
    print('Training rows:',len(rows),'over-length excluded:',skipped,flush=True)
    return Dataset.from_list(rows)


class LabelTrainer(Trainer):
    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        targets=inputs.pop('target')
        outputs=model(**inputs,use_cache=False,logits_to_keep=1)
        loss=F.cross_entropy(outputs.logits[:,-1,:].float(),targets)
        return (loss,outputs) if return_outputs else loss


def main():
    revisions=json.loads((HERE/'source_revisions.json').read_text())
    records=[json.loads(x) for x in (HERE/'data/train_v2.jsonl').read_text(encoding='utf-8').splitlines()]
    _,dev,_,_,_=prepare_data(HERE/'data',revisions,HERE/'hw5_sample_eval.csv',HERE/'kaggle_test.csv')
    model,tokenizer=load_base(revisions['model']);model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,
                                        target_modules='all-linear',bias='none',task_type='CAUSAL_LM'))
    dataset=label_dataset(records,tokenizer)
    tokenizer.padding_side='left'
    def collate(rows):
        result=tokenizer.pad([{k:r[k] for k in ['input_ids','attention_mask']} for r in rows],
                             padding=True,return_tensors='pt')
        result['target']=torch.tensor([r['target'] for r in rows],dtype=torch.long)
        return result
    output=HERE/'results_v2';output.mkdir(exist_ok=True)
    class Monitor(TrainerCallback):
        def on_log(self,args,state,control,logs=None,**kwargs):
            if logs:print('Step',state.global_step,logs,flush=True)
        def on_epoch_end(self,args,state,control,model=None,**kwargs):
            name=f'v2_epoch{round(state.epoch)}'
            model.save_pretrained(HERE/'models'/name);tokenizer.save_pretrained(HERE/'models'/name)
            for rotations in [1,5]:
                pred=score_records(model,tokenizer,dev,'boxed',rotations)
                pred.to_json(output/f'{name}_boxed_r{rotations}.json',orient='records',indent=2)
                print(name,'rotations',rotations,summarize(pred),flush=True)
            model.train();torch.cuda.empty_cache()
    args=TrainingArguments(output_dir=str(HERE/'models/v2_training'),num_train_epochs=2,
        per_device_train_batch_size=2,gradient_accumulation_steps=8,learning_rate=5e-5,
        warmup_ratio=.1,lr_scheduler_type='cosine',weight_decay=.01,max_grad_norm=1,
        optim='adamw_torch',bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        remove_unused_columns=False,logging_steps=10,save_strategy='no',report_to='none',
        seed=SEED,data_seed=SEED,dataloader_num_workers=0,disable_tqdm=True)
    trainer=LabelTrainer(model=model,args=args,train_dataset=dataset,data_collator=collate,
                          processing_class=tokenizer,callbacks=[Monitor()])
    trainer.model_accepts_loss_kwargs=False
    result=trainer.train()
    (output/'v2_training_history.json').write_text(json.dumps(trainer.state.log_history,indent=2))
    (output/'v2_training_metrics.json').write_text(json.dumps(result.metrics,indent=2))


if __name__=='__main__':main()
