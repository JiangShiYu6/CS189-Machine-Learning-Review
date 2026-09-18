"""Reproducible HW5 SFT helpers. The notebook contains the same source."""
from pathlib import Path
import gc
import hashlib
import json
import random
import re
import unicodedata
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer
from huggingface_hub import snapshot_download

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 189
LETTERS = "ABCDE"
SPECIALIZED = ["machine_learning", "college_computer_science", "college_mathematics", "high_school_statistics"]
GENERAL = ["high_school_world_history", "high_school_geography", "nutrition", "elementary_mathematics"]
SUBJECTS = SPECIALIZED + GENERAL


def fingerprint(question):
    text = unicodedata.normalize("NFKC", str(question)).casefold()
    return re.sub(r"\W+", "", text)


def read_csv(path, labeled=True):
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"id", "question", "A", "B", "C", "D", "E"}
    if labeled:
        required.add("answer")
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {required - set(frame.columns)}")
    if frame.id.duplicated().any():
        raise ValueError("Duplicate IDs")
    if labeled and not frame.answer.isin(list(LETTERS)).all():
        raise ValueError("Invalid answer labels")
    return frame


def row_record(row, group="cs189"):
    choices = [str(row[c]).strip() for c in LETTERS if str(row[c]).strip()]
    if len(choices) < 2:
        raise ValueError("A multiple-choice question needs at least two options")
    # The supplied CSVs have contiguous, nonempty choices; reject ambiguous gaps.
    if any(not str(row[c]).strip() for c in LETTERS[:len(choices)]):
        raise ValueError("Choice labels must be contiguous")
    return {"id": str(row["id"]), "question": str(row["question"]), "choices": choices,
            "answer": row.get("answer", ""), "group": group, "source": "course CSV"}


def build_prompt(record):
    options = "\n".join(f"{LETTERS[i]}. {choice}" for i, choice in enumerate(record["choices"]))
    return ("Choose exactly one correct option from the choices provided. "
            "Return only the option letter inside a LaTeX box.\n\n"
            + record["question"].strip() + "\n\n" + options)


def parse_choice_from_boxed(text):
    if not isinstance(text, str):
        return None
    boxed = re.findall(r"\\boxed\s*\{\s*([A-E])\s*\}", text, flags=re.I)
    if boxed:
        return boxed[-1].upper()
    exact = re.fullmatch(r"\s*([A-E])[.)]?\s*", text, flags=re.I)
    return exact.group(1).upper() if exact else None


def prepare_data(data_dir, revisions, eval_csv, test_csv):
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    public = [row_record(row) for row in read_csv(eval_csv).to_dict("records")]
    hidden = [row_record(row, "hidden") for row in read_csv(test_csv, False).to_dict("records")]
    forbidden = {fingerprint(r["question"]) for r in public + hidden}
    records = {"train": [], "validation": []}
    excluded = 0
    for subject in SUBJECTS:
        for split in ("test", "validation", "dev"):
            path = data_dir / f"{subject}_{split}.jsonl"
            if path.exists():
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            else:
                ds = load_dataset("cais/mmlu", subject, split=split, revision=revisions["cais/mmlu"])
                ds.to_json(str(path))
                rows = list(ds)
            destination = "train" if split == "test" else "validation"
            for index, row in enumerate(rows):
                record = {"id": f"mmlu_{subject}_{split}_{index}", "question": row["question"],
                          "choices": list(row["choices"]), "answer": LETTERS[int(row["answer"])],
                          "group": "specialized" if subject in SPECIALIZED else "general",
                          "subject": subject, "source": f"cais/mmlu:{subject}:{split}"}
                if fingerprint(record["question"]) in forbidden:
                    excluded += 1
                else:
                    records[destination].append(record)
    validation_keys = {fingerprint(r["question"]) for r in records["validation"]}
    seen = set()
    cleaned = []
    rng = random.Random(SEED)
    for subject in SUBJECTS:
        candidates = [r for r in records["train"] if r["subject"] == subject]
        rng.shuffle(candidates)
        for r in candidates[:120]:
            key = fingerprint(r["question"])
            if key not in seen and key not in validation_keys:
                cleaned.append(r)
                seen.add(key)
    rng.shuffle(cleaned)
    assert not seen.intersection(forbidden | validation_keys)
    manifest = {"model": MODEL_NAME, "revisions": revisions, "seed": SEED,
                "train_count": len(cleaned), "validation_count": len(records["validation"]),
                "public_count": len(public), "hidden_count": len(hidden),
                "excluded_test_overlap": excluded,
                "public_hidden_overlap": len({fingerprint(r['question']) for r in public} &
                                             {fingerprint(r['question']) for r in hidden}),
                "train_counts": pd.Series([r["subject"] for r in cleaned]).value_counts().to_dict(),
                "note": "MMLU test split is explicitly training data, as in the starter; dev/validation are held out."}
    return cleaned, records["validation"], public, hidden, manifest


def training_dataset(records, tokenizer):
    formatted = []
    rng = random.Random(SEED)
    for record in records:
        order = list(range(len(record["choices"])))
        rng.shuffle(order)
        original_answer = LETTERS.index(record["answer"])
        permuted = dict(record, choices=[record["choices"][i] for i in order])
        answer = LETTERS[order.index(original_answer)]
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": build_prompt(permuted)}],
                                               tokenize=False, add_generation_prompt=True)
        formatted.append({"prompt": prompt, "completion": f"\\boxed{{{answer}}}" + tokenizer.eos_token})
    return Dataset.from_list(formatted)


def load_base(revision):
    set_seed(SEED)
    snapshot = snapshot_download(MODEL_NAME, revision=revision,
                                 allow_patterns=["*.json", "*.safetensors", "*.txt"])
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(snapshot, dtype=dtype,
                                              attn_implementation="sdpa")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model, tokenizer


def predict_choices(model, tokenizer, records, batch_size=4):
    """Constrained one-token decoding after a forced boxed-answer prefix.

    Only supplied option letters are eligible. This is used identically for base,
    ablation, final model and hidden test; no labels enter the scoring path.
    """
    model.eval()
    device = next(model.parameters()).device
    letter_ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
    if not all(len(ids) == 1 for ids in letter_ids):
        raise ValueError("This decoder requires single-token answer letters")
    token_ids = [ids[0] for ids in letter_ids]
    rows = []
    tokenizer.padding_side = "left"
    for start in range(0, len(records), batch_size):
        batch = records[start:start+batch_size]
        prompts = [tokenizer.apply_chat_template([{"role":"user", "content":build_prompt(r)}],
                   tokenize=False, add_generation_prompt=True) + "\\boxed{" for r in batch]
        inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        # Match Trainer/Accelerate's mixed-precision context after adapter reload.
        with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == "cuda" and next(model.parameters()).dtype == torch.bfloat16):
            logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, token_ids].float()
        for i, r in enumerate(batch):
            scores = logits[i, :len(r["choices"])].cpu()
            prediction = LETTERS[int(scores.argmax())]
            rows.append({**r, "prediction": prediction, "output": f"\\boxed{{{prediction}}}",
                         "correct": prediction == r.get("answer", ""),
                         "choice_probabilities": torch.softmax(scores, dim=0).tolist()})
    return pd.DataFrame(rows)


def summarize(frame):
    return {group: {"correct": int(g.correct.sum()), "n": len(g), "accuracy": float(g.correct.mean())}
            for group, g in frame.groupby("group")}


def train_adapter(model, tokenizer, records, output_dir, epochs=2, learning_rate=1e-4, max_steps=-1):
    set_seed(SEED)
    tokenizer.padding_side = "right"
    model.config.use_cache = False
    config = SFTConfig(output_dir=str(output_dir), num_train_epochs=epochs, max_steps=max_steps,
                       per_device_train_batch_size=1, gradient_accumulation_steps=16,
                       learning_rate=learning_rate, warmup_ratio=0.1, lr_scheduler_type="cosine",
                       weight_decay=0.01, optim="adamw_torch", max_grad_norm=1.0,
                       logging_steps=5, save_strategy="no", report_to="none", seed=SEED,
                       data_seed=SEED, max_length=1024, packing=False, completion_only_loss=True,
                       bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
                       fp16=False, gradient_checkpointing=True,
                       gradient_checkpointing_kwargs={"use_reentrant": False},
                       dataloader_num_workers=0)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear",
                      bias="none", task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=model, args=config, train_dataset=training_dataset(records, tokenizer),
                         processing_class=tokenizer, peft_config=lora)
    result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    history = trainer.state.log_history
    (Path(output_dir)/"loss_history.json").write_text(json.dumps(history, indent=2))
    model = trainer.model
    model.eval()
    return model, history, result.metrics


def compare_examples(base, final, count=2):
    before = base.set_index("id")
    after = final.set_index("id")
    assert before.index.equals(after.index)
    general = before.group == "general"
    result = {}
    for name, mask in [("forgetting", general & before.correct & ~after.correct),
                       ("gains", general & ~before.correct & after.correct)]:
        selected = sorted(list(before.index[mask]), key=lambda idx:
                          len(before.loc[idx, 'question']) + sum(map(len, before.loc[idx, 'choices'])))[:count]
        result[name] = [{"id": idx, "question": before.loc[idx, "question"],
                         "choices": before.loc[idx, "choices"], "answer": before.loc[idx, "answer"],
                         "base_output": before.loc[idx, "output"], "final_output": after.loc[idx, "output"],
                         "source": before.loc[idx, "source"]} for idx in selected]
    return result


def save_submission(predictions, test_csv, output_path):
    ids = read_csv(test_csv, False).id.astype(str).tolist()
    submission = predictions[["id", "prediction"]].copy()
    if submission.id.astype(str).tolist() != ids or submission.id.duplicated().any():
        raise ValueError("Prediction IDs must match the test file exactly and in order")
    if not submission.prediction.isin(list(LETTERS)).all():
        raise ValueError("Invalid prediction")
    submission.to_csv(output_path, index=False)
    return submission
