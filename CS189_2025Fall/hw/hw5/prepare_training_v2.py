"""Prepare disjoint MMLU/AQuA training data without using evaluation answers.

Run after data/audit_v2.jsonl has been created. No training is performed.
The original MMLU sampling is reproduced without importing GPU libraries.
"""
import argparse
from collections import Counter
import csv
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata
import urllib.request

SEED = 189
AQUA_REVISION = "26b64ed1f22208c6742f47df547dfd65088ec30d"
AQUA_URL = f"https://raw.githubusercontent.com/google-deepmind/AQuA/{AQUA_REVISION}/train.json"
SPECIALIZED = ["machine_learning", "college_computer_science", "college_mathematics", "high_school_statistics"]
GENERAL = ["high_school_world_history", "high_school_geography", "nutrition", "elementary_mathematics"]
LETTERS = "ABCDE"


def fingerprint(question):
    return re.sub(r"\W+", "", unicodedata.normalize("NFKC", str(question)).casefold())


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def course_questions(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [row["question"] for row in csv.DictReader(stream)]


def original_training(data, course_keys):
    """Match prepare_data's original sampling order, seed, and cap exactly."""
    raw_train, validation = [], []
    for subject in SPECIALIZED + GENERAL:
        for split in ("test", "validation", "dev"):
            for index, row in enumerate(jsonl(data / f"{subject}_{split}.jsonl")):
                if fingerprint(row["question"]) in course_keys:
                    continue
                record = {"id": f"mmlu_{subject}_{split}_{index}", "question": row["question"],
                          "choices": list(row["choices"]), "answer": LETTERS[int(row["answer"])],
                          "group": "specialized" if subject in SPECIALIZED else "general",
                          "subject": subject, "source": f"cais/mmlu:{subject}:{split}"}
                (raw_train if split == "test" else validation).append(record)
    validation_keys = {fingerprint(row["question"]) for row in validation}
    seen, cleaned = set(), []
    rng = random.Random(SEED)
    for subject in SPECIALIZED + GENERAL:
        candidates = [r for r in raw_train if r["subject"] == subject]
        rng.shuffle(candidates)
        for record in candidates[:120]:
            key = fingerprint(record["question"])
            if key not in seen and key not in validation_keys:
                seen.add(key)
                cleaned.append(record)
    rng.shuffle(cleaned)
    return cleaned, validation


def near_course(key, course_keys):
    for other in course_keys:
        # This is a safe upper bound on SequenceMatcher's ratio.
        if 2 * min(len(key), len(other)) / max(1, len(key) + len(other)) < 0.9:
            continue
        if SequenceMatcher(None, key, other, autojunk=False).ratio() >= 0.9:
            return True
    return False


def prepare(root, proxy):
    data = root / "data"
    audit_path = data / "audit_v2.jsonl"
    if not audit_path.exists():
        raise FileNotFoundError("Create data/audit_v2.jsonl before preparing training data.")
    audit = jsonl(audit_path)
    course = course_questions(root / "hw5_sample_eval.csv") + course_questions(root / "kaggle_test.csv")
    course_keys = {fingerprint(question) for question in course}
    original, validation = original_training(data, course_keys)
    old_manifest = json.loads((root / "results" / "manifest.json").read_text(encoding="utf-8"))
    assert len(original) == old_manifest["train_count"] == 907
    assert len(validation) == old_manifest["validation_count"] == 217
    forbidden = course_keys | {fingerprint(r["question"]) for r in validation + audit}
    excluded = Counter()
    seen = set()

    def accept(record, source):
        key = fingerprint(record["question"])
        if key in forbidden:
            excluded[f"{source}_heldout_exact"] += 1
            return False
        if key in seen:
            excluded[f"{source}_duplicate"] += 1
            return False
        if near_course(key, course_keys):
            excluded[f"{source}_course_near_duplicate"] += 1
            return False
        seen.add(key)
        return True

    specialized = [r for r in original if r["group"] == "specialized" and accept(r, "mmlu")]
    general_pool = [r for r in original if r["group"] == "general" and accept(r, "mmlu")]
    rng = random.Random(SEED)
    rng.shuffle(general_pool)
    if len(general_pool) < 150:
        raise ValueError("Fewer than 150 clean general examples remain.")
    general = general_pool[:150]
    # Only examples actually selected are duplicate exclusions for AQuA.
    seen = {fingerprint(r["question"]) for r in specialized + general}
    aqua_path = data / "aqua_train.json"
    if not aqua_path.exists():
        print(f"Downloading {AQUA_URL}", flush=True)
        opener = (urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
                  if proxy else urllib.request.build_opener())
        with opener.open(AQUA_URL, timeout=180) as response:
            payload = response.read()
        temporary = aqua_path.with_suffix(".download")
        temporary.write_bytes(payload)
        temporary.replace(aqua_path)
    aqua_sha = hashlib.sha256(aqua_path.read_bytes()).hexdigest()
    raw_aqua = jsonl(aqua_path)
    rng.shuffle(raw_aqua)
    aqua = []
    for raw in raw_aqua:
        options = raw.get("options", [])
        answer = raw.get("correct", "").strip().upper()
        if len(options) != 5 or answer not in LETTERS:
            excluded["aqua_invalid"] += 1
            continue
        choices = []
        for letter, option in zip(LETTERS, options):
            match = re.fullmatch(r"\s*" + letter + r"\s*[).:]\s*(.+)", str(option), re.S)
            if not match:
                break
            choices.append(match.group(1).strip())
        question = str(raw.get("question", "")).strip()
        if not question or len(choices) != 5 or any(not choice for choice in choices):
            excluded["aqua_invalid"] += 1
            continue
        if len(question) + sum(map(len, choices)) > 1800:
            excluded["aqua_too_long"] += 1
            continue
        key = fingerprint(question)
        record = {"id": "aqua_train_" + hashlib.sha256(key.encode()).hexdigest()[:16],
                  "question": question, "choices": choices, "answer": answer,
                  "group": "specialized", "subject": "algebra_word_problems",
                  "source": f"google-deepmind/AQuA:{AQUA_REVISION}:train"}
        if accept(record, "aqua"):
            aqua.append(record)
        if len(aqua) == 400:
            break
    if len(aqua) != 400:
        raise ValueError(f"Only {len(aqua)} valid AQuA records available.")
    combined = specialized + general + aqua
    rng.shuffle(combined)
    assert len({r["id"] for r in combined}) == len(combined)
    assert not {fingerprint(r["question"]) for r in combined} & forbidden
    assert not any(near_course(fingerprint(r["question"]), course_keys) for r in combined)
    output = data / "train_v2.jsonl"
    output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in combined), encoding="utf-8")
    manifest = {
        "seed": SEED, "original_mmlu_count": len(original),
        "original_mmlu_revisions": old_manifest["revisions"],
        "original_specialized_count": sum(r["group"] == "specialized" for r in original),
        "selected_counts": {"mmlu_specialized": len(specialized), "mmlu_general": len(general), "aqua": len(aqua)},
        "train_count": len(combined), "subject_counts": dict(Counter(r["subject"] for r in combined)),
        "answer_counts": dict(Counter(r["answer"] for r in combined)),
        "aqua": {"repository": "https://github.com/google-deepmind/AQuA", "revision": AQUA_REVISION,
                 "url": AQUA_URL, "download_sha256": aqua_sha, "raw_rows": len(raw_aqua),
                 "max_question_and_choices_characters": 1800, "rationales_used": False},
        "heldout_counts": {"original_validation": len(validation), "audit": len(audit), "course_rows": len(course)},
        "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "train_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "excluded": dict(excluded),
        "decontamination": "NFKC casefold nonword-stripped question exact match against course, original validation, audit; SequenceMatcher(autojunk=False) >= 0.9 against all course questions; exact training deduplication.",
        "sampling": "Reproduce original MMLU sample; keep clean specialized; seed-189 shuffle general then take 150; continue RNG to shuffle official AQuA train and take first 400 legal clean short records; shuffle combined.",
        "note": "Evaluation answers are never used. AQuA rationales are never used. Course and audit questions are used solely for overlap exclusion."
    }
    results = root / "results_v2"
    results.mkdir(exist_ok=True)
    (results / "training_data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--proxy", default=None, help="Optional HTTP proxy for the public data download")
    args = parser.parse_args()
    prepare(args.root, args.proxy)
