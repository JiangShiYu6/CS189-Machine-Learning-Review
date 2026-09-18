"""Freeze a fresh, label-independent audit sample before evaluating v2 models."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import random

from sft_pipeline import GENERAL, LETTERS, fingerprint, prepare_data

HERE = Path(__file__).resolve().parent
AUDIT_SEED = 20260918


def main():
    revisions = json.loads((HERE / "source_revisions.json").read_text(encoding="utf-8"))
    train, development, public, hidden, _ = prepare_data(
        HERE / "data", revisions, HERE / "hw5_sample_eval.csv", HERE / "kaggle_test.csv")
    excluded = {fingerprint(row["question"]) for row in train + development + public + hidden}
    rng = random.Random(AUDIT_SEED)
    chosen = []
    seen = set(excluded)
    candidate_counts = {}
    source_hashes = {}
    # Other original specialized subjects have no unused test examples left.
    # Sampling never accesses the answer field; it is copied only after selection.
    quotas = {"high_school_statistics": 80, **{subject: 20 for subject in GENERAL}}
    for subject, quota in quotas.items():
        path = HERE / "data" / f"{subject}_test.jsonl"
        raw = path.read_bytes()
        source_hashes[path.name] = hashlib.sha256(raw).hexdigest()
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        eligible = []
        subject_seen = set(seen)
        for index, row in enumerate(rows):
            key = fingerprint(row["question"])
            if key not in subject_seen:
                eligible.append((index, row))
                subject_seen.add(key)
        candidate_counts[subject] = len(eligible)
        rng.shuffle(eligible)
        for index, row in eligible[:quota]:
            seen.add(fingerprint(row["question"]))
            chosen.append({
                "id": f"mmlu_{subject}_test_{index}",
                "question": row["question"], "choices": list(row["choices"]),
                "answer": LETTERS[int(row["answer"])],
                "group": "general" if subject in GENERAL else "specialized",
                "subject": subject, "source": f"cais/mmlu:{subject}:test",
            })
    keys = [fingerprint(row["question"]) for row in chosen]
    assert len(keys) == len(set(keys))
    assert not set(keys).intersection(excluded)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in chosen).encode("utf-8")
    manifest = {
        "seed": AUDIT_SEED,
        "dataset_revision": revisions["cais/mmlu"],
        "selection": "Fixed-seed random selection from unused cached test records, without inspecting gold labels or predictions.",
        "scope_limit": "Specialized audit covers high_school_statistics only; it is not a broad CS189 or machine-learning benchmark. General audit covers four previously used subject domains with new examples.",
        "prior_development_limit": "The original 217 held-out examples have already been observed and are development data for v2; these audit IDs were absent from original training and development.",
        "usage": "Freeze before model evaluation; exclude these question fingerprints from all future training and prompt demonstrations. Evaluate only after selecting the v2 method on development data.",
        "exclusion_counts": {"old_train": len(train), "old_development": len(development), "course_public": len(public), "course_hidden": len(hidden)},
        "quotas": quotas,
        "available_candidates": candidate_counts,
        "counts_by_subject": dict(Counter(row["subject"] for row in chosen)),
        "counts_by_group": dict(Counter(row["group"] for row in chosen)),
        "count": len(chosen),
        "selected_ids": [row["id"] for row in chosen],
        "question_fingerprints": keys,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "source_file_sha256": source_hashes,
    }
    output = HERE / "data" / "audit_v2.jsonl"
    manifest_path = HERE / "results_v2" / "audit_manifest.json"
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    # A rerun may confirm this split, but must never silently replace it.
    for path, expected in [(output, payload), (manifest_path, manifest_bytes)]:
        if path.exists() and path.read_bytes() != expected:
            raise RuntimeError(f"Refusing to overwrite a different frozen audit artifact: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    manifest_path.write_bytes(manifest_bytes)
    print(json.dumps({"count": len(chosen), "counts_by_group": manifest["counts_by_group"], "sha256": manifest["sha256"]}))


if __name__ == "__main__":
    main()
