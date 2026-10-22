#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build frozen target-conditioned Qwen3 embeddings for MT-CSD.

Run from MT-CSD: python preprocess/build_embedding_qwen_target_final.py --model_dir /path/to/qwen3-embedding-8b
Use --dry_run to validate inputs and inspect prompts without loading torch/model.
Outputs embeddings.npy + sentence.csv for all utterances (including provided
user history), plus train/dev/test.npy and *_ids.npy aligned to source records.
valid.json is mapped to dev. No labels are used in encoding or deduplication.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
DEFAULT_DATA = Path(__file__).resolve().parents[1] / "dataset"
# Preserve the original ZS-CSD target-final prompt for comparable experiments.
PROMPT_HEADER = ("Represent the stance expressed by the final utterance toward the target, "
                 "considering the entire conversation context.")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_dir", default="/home/xiongjie/data/models/qwen3-embedding-8b")
    p.add_argument("--data_dir", default=str(DEFAULT_DATA))
    p.add_argument("--output_dir", default="",
                   help="default: <data_dir>/qwen3_embedding_8b_target_final")
    for split in ("train", "dev", "test"):
        aliases = [f"--{split}_file"] + (["--valid_file"] if split == "dev" else [])
        p.add_argument(*aliases, dest=f"{split}_file", default="",
                       help="explicit source path; relative paths resolve from data_dir")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--device", default="auto")
    p.add_argument("--device_map", choices=["none", "auto"], default="none")
    p.add_argument("--normalize", dest="normalize", action="store_true")
    p.add_argument("--no-normalize", dest="normalize", action="store_false")
    p.add_argument("--trust_remote_code", dest="trust_remote_code", action="store_true")
    p.add_argument("--no-trust_remote_code", dest="trust_remote_code", action="store_false")
    p.add_argument("--dry_run", action="store_true")
    p.set_defaults(normalize=True, trust_remote_code=True)
    return p


def target_final_prompt(record):
    sentences = record.get("sentences")
    if not isinstance(sentences, list) or not sentences:
        raise ValueError("record must have a nonempty sentences list")
    target, text = record.get("target"), sentences[-1]
    if not isinstance(target, str) or not target.strip():
        raise ValueError("target must be a nonempty string")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("utterance must be a nonempty string")
    return (f"{PROMPT_HEADER}\n\nTarget:\n{target.strip()}\n\n"
            f"Conversation:\n[1] {text}\n\nFinal utterance:\n{text}")


def resolve_split_files(cfg):
    paths = {}
    root = Path(cfg.data_dir)
    for split in ("train", "dev", "test"):
        given = getattr(cfg, f"{split}_file")
        if given:
            path = Path(given)
            if not path.is_absolute():
                path = root / path
        else:
            names = (["valid.json", "dev.json", "valid_data.json", "dev_data.json"]
                     if split == "dev" else [f"{split}.json", f"{split}_data.json"])
            path = next((root / name for name in names if (root / name).is_file()), None)
        if path is None or not path.is_file():
            raise FileNotFoundError(f"{split}: source file not found under {root}: {given}")
        paths[split] = path
    if len({p.resolve() for p in paths.values()}) != 3:
        raise ValueError("train/dev/test must use different source files")
    return paths


def prepare(cfg):
    paths = resolve_split_files(cfg)
    rows, key_to_gid, split_ids, final_gids, stats = [], {}, {}, {}, {}
    for split, path in paths.items():
        records = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(records, list) or not records:
            raise ValueError(f"{path}: expected a nonempty record list")
        ids, gids = [], []
        dialog_count = user_count = 0
        for i, rec in enumerate(records):
            try:
                target_final_prompt(rec)
                rid = rec.get("id", rec.get("doc_id"))
                if type(rid) not in (int, str):
                    raise ValueError("id/doc_id must be an integer or string")
                ids.append(rid)
                sentences = rec["sentences"]
                history = rec.get("last_speaker_sentences", [])
                if not isinstance(history, list):
                    raise ValueError("last_speaker_sentences must be a list")
                for text in sentences + history:
                    # Validate every historical utterance, not only the final one.
                    target_final_prompt({"target": rec["target"], "sentences": [text]})
                    key = (rec["target"], text)
                    if key not in key_to_gid:
                        key_to_gid[key] = len(rows)
                        rows.append(key)
                gids.append(key_to_gid[(rec["target"], sentences[-1])])
                dialog_count += len(sentences)
                user_count += len(history)
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{split} row={i}: {exc}") from exc
        if len(set(ids)) != len(ids) or len({type(x) for x in ids}) != 1:
            raise ValueError(f"{split}: IDs must be unique and have a consistent type")
        split_ids[split] = np.asarray(ids)
        final_gids[split] = np.asarray(gids, dtype=np.int64)
        stats[split] = {"source_json": str(path.resolve()), "records": len(records),
                        "dialogue_utterances": dialog_count, "provided_user_utterances": user_count,
                        "prompt_example": target_final_prompt(records[0])}
    return rows, split_ids, final_gids, stats


def load_model(cfg):
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not Path(cfg.model_dir).is_dir():
        raise FileNotFoundError(f"model directory not found: {cfg.model_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else torch.device(cfg.device)
    tok = AutoTokenizer.from_pretrained(cfg.model_dir, trust_remote_code=cfg.trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    kw = {"trust_remote_code": cfg.trust_remote_code}
    if cfg.dtype != "auto":
        kw["torch_dtype"] = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg.dtype]
    if cfg.device_map == "auto":
        kw["device_map"] = "auto"
    model = AutoModel.from_pretrained(cfg.model_dir, **kw).eval()
    if cfg.device_map == "none":
        model.to(device)
    return tok, model, model.get_input_embeddings().weight.device


def encode(rows, tok, model, device, cfg):
    import torch
    import torch.nn.functional as F

    chunks = []
    truncated = 0
    with torch.inference_mode():
        for start in range(0, len(rows), cfg.batch_size):
            texts = [target_final_prompt({"target": t, "sentences": [s]})
                     for t, s in rows[start:start + cfg.batch_size]]
            lengths = tok(texts, padding=False, truncation=False, return_length=True)["length"]
            truncated += sum(n > cfg.max_length for n in lengths)
            batch = tok(texts, padding=True, truncation=True, max_length=cfg.max_length,
                        return_tensors="pt").to(device)
            # Left padding makes the final token the last non-padding token.
            emb = model(**batch).last_hidden_state[:, -1].float()
            if cfg.normalize:
                emb = F.normalize(emb, p=2, dim=1)
            chunks.append(emb.cpu().numpy())
            if start == 0 or (start // cfg.batch_size + 1) % 20 == 0:
                logger.info("Encoded %d/%d unique target+utterance pairs", min(start + cfg.batch_size, len(rows)), len(rows))
    return np.concatenate(chunks, axis=0), truncated


def run(cfg):
    if cfg.batch_size <= 0 or cfg.max_length <= 0:
        raise ValueError("batch_size and max_length must be positive")
    rows, ids, final_gids, stats = prepare(cfg)
    summary = {"mode": "target_final", "config": vars(cfg).copy(),
               "prompt_header": PROMPT_HEADER, "unique_target_utterances": len(rows),
               "splits": stats}
    logger.info("Validated %d unique target+utterance pairs; splits=%s", len(rows),
                {s: x["records"] for s, x in stats.items()})
    if cfg.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    tok, model, device = load_model(cfg)
    embeddings, truncated = encode(rows, tok, model, device, cfg)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(rows) or not np.isfinite(embeddings).all():
        raise ValueError("Invalid embedding matrix shape or non-finite values")
    out = Path(cfg.output_dir or Path(cfg.data_dir) / "qwen3_embedding_8b_target_final")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", embeddings, allow_pickle=False)
    with (out / "sentence.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["global_id", "target", "sentence"])
        writer.writerows((gid, t, s) for gid, (t, s) in enumerate(rows))
    for split in stats:
        np.save(out / f"{split}.npy", embeddings[final_gids[split]], allow_pickle=False)
        np.save(out / f"{split}_ids.npy", ids[split], allow_pickle=False)
        np.save(out / f"{split}_global_ids.npy", final_gids[split], allow_pickle=False)
        stats[split]["shape"] = [len(ids[split]), embeddings.shape[1]]
    summary.update(embedding_shape=list(embeddings.shape), truncated_prompts=truncated)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved embeddings to %s; truncated prompts=%d", out, truncated)
    return summary


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = build_parser().parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
