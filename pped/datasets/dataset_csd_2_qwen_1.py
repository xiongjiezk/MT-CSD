"""Frozen target-conditioned Qwen features for PPED; no tokenizer is loaded.

Two supported stores:
1. embeddings.npy + sentence.csv (global_id,target,sentence).
2. <split>.npy + <split>_ids.npy from build_embedding_qwen_target_final.py.
   In case 2, historical utterances must appear as final utterances of other
   records with the SAME target in this split. Matching never uses stance labels.
"""
import csv
from collections import defaultdict, Counter
import re
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class DialogueStanceDataset(Dataset):
    def __init__(self, json_path, embedding_path, debug=False, shuffle_data=False,
                 seed=1234, sentence_csv="", ids_path="", user_history_mode="prefix", preload_embeddings=True, include_history=True, enable_pairs=None):
        # Existing entry points do not pass a split. Recognize standard training
        # filenames; callers with other names must explicitly use enable_pairs=True.
        if enable_pairs is None:
            enable_pairs = bool(re.search(r"(^|[_\-.])train(?:ing)?($|[_\-.])",
                                          Path(json_path).stem.lower()))
        self.enable_pairs = bool(enable_pairs)
        if user_history_mode not in ("prefix", "provided"):
            raise ValueError("user_history_mode must be prefix or provided")
        records = json.loads(Path(json_path).read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError("JSON must contain a list of records")
        matrix = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
        if matrix.ndim != 2 or matrix.shape[1] == 0:
            raise ValueError("embeddings must have shape [N,D]")
        self.embedding_dim = int(matrix.shape[1])
        if preload_embeddings:
            matrix = np.array(matrix, dtype=np.float32, copy=True)
            if not np.isfinite(matrix).all():
                raise ValueError("Embedding contains NaN/Inf")
        self.matrix = matrix
        self.preloaded = preload_embeddings
        by_key = {}
        row_aligned = not sentence_csv
        if sentence_csv:
            with open(sentence_csv, encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                if not {"global_id", "target", "sentence"}.issubset(reader.fieldnames or []):
                    raise ValueError("Need target-conditioned CSV: global_id,target,sentence. "
                                     "Old isolated sentence-only features are not valid here.")
                seen = set()
                for r in reader:
                    gid = int(r["global_id"])
                    key = (r["target"], r["sentence"])
                    if gid in seen or not 0 <= gid < len(matrix) or key in by_key:
                        raise ValueError("duplicate/out-of-range mapping row or duplicate target+sentence")
                    seen.add(gid)
                    by_key[key] = gid
                if len(seen) != len(matrix):
                    raise ValueError("CSV and embedding row counts differ")
        else:
            ids_path = ids_path or str(Path(embedding_path).with_name(Path(embedding_path).stem + "_ids.npy"))
            ids = np.load(ids_path, allow_pickle=False)
            expected = [r.get("id", r.get("doc_id")) for r in records]
            if (ids.ndim != 1 or len(matrix) != len(records) or ids.tolist() != expected
                    or None in expected or len(set(expected)) != len(expected)):
                raise ValueError("Embedding IDs/order must exactly match unique JSON record IDs")
            for row, r in enumerate(records):
                if not r.get("sentences"):
                    raise ValueError(f"Empty conversation at row {row}")
                by_key.setdefault((r["target"], r["sentences"][-1]), row)
        self.data = []
        for row, r in enumerate(records[:100] if debug else records):
            sentences, speakers = r["sentences"], r["speakers"]
            if not sentences or len(sentences) != len(speakers):
                raise ValueError(f"Invalid sentences/speakers at row {row}")
            labels = r.get("all_labels")
            if labels is not None and (len(labels) != len(sentences) or any(x not in (0, 1, 2) for x in labels)):
                raise ValueError(f"Invalid all_labels at row {row}")
            label = r.get("label")
            if label is not None and label not in (0, 1, 2):
                raise ValueError(f"Invalid final label at row {row}")
            if labels is not None and label is not None and labels[-1] != label:
                raise ValueError(f"Final label disagrees with all_labels at row {row}")
            if not include_history:
                sentences, speakers = sentences[-1:], speakers[-1:]
                labels = labels[-1:] if labels is not None else None
            def lookup(text):
                key = (r["target"], text)
                if key not in by_key:
                    raise ValueError(f"Missing target+sentence embedding for record {r.get('id')}: {key!r}. "
                                     "Provide a complete target-conditioned sentence CSV/matrix.")
                return by_key[key]
            gids = [lookup(s) for s in sentences]
            if row_aligned:
                gids[-1] = row  # preserve exact baseline final vector, even for duplicate text
            if not include_history:
                user_gids, user_labels = [], []
            elif user_history_mode == "prefix":
                positions = [j for j in range(len(sentences)-1) if speakers[j] == speakers[-1]]
                user_gids = [gids[j] for j in positions]
                user_labels = [labels[j] for j in positions] if labels is not None else None
            else:
                user_gids = [lookup(s) for s in r.get("last_speaker_sentences", [])]
                user_labels = r.get("last_speaker_sentences_labels")
                if user_labels is not None and (len(user_labels) != len(user_gids) or any(x not in (0,1,2) for x in user_labels)):
                    raise ValueError(f"Invalid provided user-history labels at row {row}")
            self.data.append(dict(doc_id=r.get("id", r.get("doc_id")), gids=gids,
                                  user_gids=user_gids, speakers=speakers, label=label,
                                  all_labels=labels, target=r["target"], target_type=r.get("target_type", ""),
                                  last_speaker_sentences_labels=user_labels,
                                  _pair_key=(tuple(r["sentences"]), tuple(r["speakers"])),
                                  _user_texts=tuple(r.get("last_speaker_sentences", []))
                                      if include_history and user_history_mode == "provided" else ()))
        if shuffle_data:
            random.Random(seed).shuffle(self.data)
        # Build AFTER shuffling; never search another split, or use labels to
        # choose between several possible counterparts. Ambiguous groups are skipped.
        self.dual_indices = [-1] * len(self.data)
        self.dual_user_aligned = [False] * len(self.data)
        stats = Counter()
        if self.enable_pairs:
            groups = defaultdict(list)
            for i, item in enumerate(self.data):
                groups[item['_pair_key']].append(i)
            for indices in groups.values():
                reason = None
                if len(indices) != 2:
                    reason = 'missing' if len(indices) == 1 else 'ambiguous'
                else:
                    a, b = (self.data[i] for i in indices)
                    if a['target'] == b['target']:
                        reason = 'same_target'
                    elif a['label'] is None or b['label'] != {0: 1, 1: 0, 2: 2}[a['label']]:
                        reason = 'incompatible_labels'
                if reason:
                    stats[reason] += len(indices)
                else:
                    i, j = indices
                    self.dual_indices[i], self.dual_indices[j] = j, i
                    aligned = a['_user_texts'] == b['_user_texts']
                    self.dual_user_aligned[i] = self.dual_user_aligned[j] = aligned
                    stats['paired'] += 2
                    stats['paired_user_aligned' if aligned else 'paired_dialog_only'] += 2
        self.pair_stats = dict(stats)
        for item in self.data:
            item.pop('_pair_key')
            item.pop('_user_texts')
        logging.info("Qwen dual targets: enabled=%s, rows=%d, coverage=%s",
                     self.enable_pairs, len(self.data), dict(stats))
        logging.info("Qwen PPED: %d records, dim=%d, user_history=%s, store=%s",
                     len(self.data), self.embedding_dim, user_history_mode, embedding_path)

    def __len__(self):
        return len(self.data)

    def _materialize(self, idx):
        item = dict(self.data[idx])
        for key, out in (("gids", "utterance_embeddings"), ("user_gids", "user_embeddings")):
            vectors = np.array(self.matrix[item.pop(key)], dtype=np.float32, copy=True)
            if not self.preloaded and not np.isfinite(vectors).all():
                raise ValueError("Embedding contains NaN/Inf")
            item[out] = torch.from_numpy(vectors)
        return item

    def __getitem__(self, idx):
        item = self._materialize(idx)
        if not self.enable_pairs:
            return item
        counterpart = self.dual_indices[idx]
        item['dual_mask'] = counterpart >= 0
        item['dual_user_aligned'] = self.dual_user_aligned[idx]
        if counterpart >= 0:
            dual = self._materialize(counterpart)
            item['dual_utterance_embeddings'] = dual['utterance_embeddings']
            # Reuse source flattened user indices only for identical candidates.
            # Otherwise the auxiliary user pool is disabled, NOT filled with
            # invented target-conditioned examples or another speaker's history.
            item['dual_user_embeddings'] = (dual['user_embeddings'] if self.dual_user_aligned[idx]
                                            else torch.zeros_like(item['user_embeddings']))
            item['dual_label'] = dual['label']
        else:
            # Same shapes preserve the original flattened indices. These rows
            # are excluded from the auxiliary loss, including their final vector.
            item['dual_utterance_embeddings'] = torch.zeros_like(item['utterance_embeddings'])
            item['dual_user_embeddings'] = torch.zeros_like(item['user_embeddings'])
            item['dual_label'] = -100
        return item


def dialogue_collate_fn(batch):
    if not batch:
        raise ValueError("Empty batch")
    result = {k: [b[k] for b in batch] for k in
              ("doc_id", "speakers", "target", "target_type", "last_speaker_sentences_labels")}
    for key, index in (("utterance_embeddings", "dia_idx"), ("user_embeddings", "user_history_idx")):
        offset, spans = 0, []
        for b in batch:
            spans.append((offset, offset + len(b[key])))
            offset += len(b[key])
        result[key] = torch.cat([b[key] for b in batch], dim=0)
        result[index] = spans
    for key in ("label", "all_labels"):
        values = [b[key] for b in batch]
        if any(v is None for v in values):
            if not all(v is None for v in values):
                raise ValueError(f"Mixed missing/present {key} in a batch")
        else:
            result[key] = torch.tensor(values, dtype=torch.long) if key == "label" else values
    # Prepare small index/label tensors on CPU once, instead of per sample on GPU.
    result["final_indices"] = torch.tensor([ed-1 for st, ed in result["dia_idx"]])
    for prefix, spans, is_dialog in (("dialog", result["dia_idx"], True),
                                     ("user", result["user_history_idx"], False)):
        lengths = [ed-st-int(is_dialog) for st, ed in spans]
        width = max(1, max(lengths))
        indices = torch.zeros((len(batch), width), dtype=torch.long)
        valid = torch.zeros_like(indices, dtype=torch.bool)
        types = torch.zeros_like(indices)
        labels = torch.zeros_like(indices)
        for i, ((st, ed), length) in enumerate(zip(spans, lengths)):
            indices[i, :length] = torch.arange(st, st+length)
            valid[i, :length] = True
            if is_dialog:
                types[i, :length] = torch.tensor([sp != batch[i]["speakers"][-1] for sp in batch[i]["speakers"][:-1]], dtype=torch.long)
                source_labels = batch[i]["all_labels"]
            else:
                source_labels = batch[i]["last_speaker_sentences_labels"]
            if source_labels is not None:
                labels[i, :length] = torch.tensor(source_labels[:length], dtype=torch.long)
        result[prefix+"_indices"] = indices
        result[prefix+"_valid"] = valid
        result[prefix+"_speaker_types"] = types
        result[prefix+"_labels"] = labels
    if any('dual_mask' in b for b in batch):
        if not all('dual_mask' in b for b in batch):
            raise ValueError("Do not mix pair-enabled and ordinary datasets in one batch")
        for b in batch:
            for key in ('utterance_embeddings', 'user_embeddings'):
                if b['dual_' + key].shape != b[key].shape:
                    raise ValueError("Dual history candidates must align with source candidates")
            if not b['dual_mask'] and b['dual_label'] != -100:
                raise ValueError("Missing dual must use ignore label -100")
        result['dual_mask'] = torch.tensor([b['dual_mask'] for b in batch], dtype=torch.bool)
        result['dual_user_aligned'] = torch.tensor([b['dual_user_aligned'] for b in batch], dtype=torch.bool)
        result['dual_labels'] = torch.tensor([b['dual_label'] for b in batch], dtype=torch.long)
        for key in ('utterance_embeddings', 'user_embeddings'):
            result['dual_' + key] = torch.cat([b['dual_' + key] for b in batch], dim=0)
    return result
