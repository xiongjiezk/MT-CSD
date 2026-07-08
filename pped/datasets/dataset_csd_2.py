# !/usr/bin/python
# -*- coding: utf-8 -*-
# @author: xiongjie <xiongjiezk@163.com>
# @date: 2026/5/15

"""Template-based dialogue stance dataset.

This dataset follows the baseline's target-aware template style, while keeping
our extra fields for latent evidence modeling:
- all_labels
- last_speaker_sentences
- last_speaker_sentences_labels

The goal is to make utterance/target interaction explicit at the input level,
then let the model learn a latent evidence distribution over candidate history
sentences.
"""

from collections import Counter
import json
import logging
import random
import time
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class DialogueStanceDataset(Dataset):
    def __init__(self, json_path, tokenizer_dir, debug=False, shuffle_data=False, seed=1234):
        start_time = time.time()
        with open(json_path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)
        if debug:
            self.raw_data = self.raw_data[:100]
        logging.info(f"load json data from {json_path}, size {len(self.raw_data)}, cost {time.time() - start_time:.3f}s")

        start_time = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        logging.info(f"load tokenizer from {tokenizer_dir} done, cost {time.time() - start_time:.3f}s")

        self.shuffle_data = shuffle_data
        self.seed = seed
        self.rng = random.Random(self.seed)
        np.random.seed(self.seed)

        start_time = time.time()
        self.data = self._build_examples(self.raw_data)
        if self.shuffle_data:
            self.rng.shuffle(self.data)
        label_counts = Counter([x["label"] for x in self.data])
        logging.info(f"build examples done, cost {time.time() - start_time:.3f}s, shuffle_data={self.shuffle_data}, length={len(self.data)}, label_counts={label_counts}")

    def _encode_text(self, text: str):
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            padding=False,
            return_attention_mask=True,
            return_token_type_ids=True,
        )
        token_type_ids = encoded.get("token_type_ids")
        if token_type_ids is None:
            token_type_ids = [0] * len(encoded["input_ids"])
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "token_type_ids": token_type_ids,
        }

    @staticmethod
    def _normalize_target_type(target_type):
        return target_type if target_type is not None else ""

    def _build_utterance_text(self, speaker, utterance, target, target_type):
        type_hint = f"[{target_type}]" if target_type else ""
        # Baseline-style explicit target-conditioned template.
        return f"[CLS]In the utterance \"{utterance}\", the user {speaker}{type_hint} expresses a stance towards [SEP]{target}[SEP]"
   

    def _build_examples(self, raw_data: List[Dict[str, Any]]):
        examples = []
        for dialogue in raw_data:
            target = dialogue["target"]
            target_type = self._normalize_target_type(dialogue.get("target_type", ""))
            speakers = dialogue["speakers"]
            sentences = dialogue["sentences"]
            all_labels = dialogue["all_labels"]
            last_label = dialogue["label"]
            last_speaker_sentences = dialogue.get("last_speaker_sentences", [])
            last_speaker_sentences_labels = dialogue.get("last_speaker_sentences_labels", [])

            if len(all_labels) != len(sentences):
                raise ValueError(
                    f"all_labels length ({len(all_labels)}) must match sentences length ({len(sentences)}) for dialogue id={dialogue.get('id', dialogue.get('doc_id'))}"
                )
            if len(sentences) != len(speakers):
                raise ValueError(
                    f"sentences length ({len(sentences)}) must match speakers length ({len(speakers)}) for dialogue id={dialogue.get('id', dialogue.get('doc_id'))}"
                )

            if len(last_speaker_sentences) != len(last_speaker_sentences_labels):
                raise ValueError(
                    f"last_speaker_sentences length ({len(last_speaker_sentences)}) must match last_speaker_sentences_labels length ({len(last_speaker_sentences_labels)}) for dialogue id={dialogue.get('id', dialogue.get('doc_id'))}"
                )

            utterance_inputs = []
            for spk, utt in zip(speakers, sentences):
                text = self._build_utterance_text(spk, utt, target, target_type)
                utterance_inputs.append(self._encode_text(text))

            last_speaker_sentences_inputs = []
            last_speaker = speakers[-1] if len(speakers) > 0 else ""
            for utt in last_speaker_sentences:
                text = self._build_utterance_text(last_speaker, utt, target, target_type)
                last_speaker_sentences_inputs.append(self._encode_text(text))

            target_inputs = self._encode_text(f"[CLS]{target}[SEP]")

            examples.append(
                {
                    "doc_id": dialogue.get("id", dialogue.get("doc_id")),
                    "input_ids": [x["input_ids"] for x in utterance_inputs],
                    "attention_mask": [x["attention_mask"] for x in utterance_inputs],
                    "token_type_ids": [x["token_type_ids"] for x in utterance_inputs],
                    "speakers": speakers,
                    "label": last_label,
                    "all_labels": all_labels,
                    "target": target,
                    "target_type": target_type,
                    "last_speaker_sentences_input_ids": [x["input_ids"] for x in last_speaker_sentences_inputs],
                    "last_speaker_sentences_attention_mask": [x["attention_mask"] for x in last_speaker_sentences_inputs],
                    "last_speaker_sentences_token_type_ids": [x["token_type_ids"] for x in last_speaker_sentences_inputs],
                    "last_speaker_sentences_labels": last_speaker_sentences_labels,
                    "last_speaker_sentences_count": len(last_speaker_sentences_inputs),
                    "target_input_ids": target_inputs["input_ids"],
                    "target_attention_mask": target_inputs["attention_mask"],
                    "target_token_type_ids": target_inputs["token_type_ids"],
                }
            )
        return examples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def _pad_sequence(seq, max_len, pad_value=0):
    return seq + [pad_value] * (max_len - len(seq))


def dialogue_collate_fn(batch):

    input_ids = [b["input_ids"] for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]
    token_type_ids = [b["token_type_ids"] for b in batch]
    speakers = [b["speakers"] for b in batch]
    labels = [b["label"] for b in batch]
    all_labels = [b["all_labels"] for b in batch]
    targets = [b["target"] for b in batch]
    target_types = [b.get("target_type", "") for b in batch]
    last_speaker_sentences_input_ids = [b["last_speaker_sentences_input_ids"] for b in batch]
    last_speaker_sentences_attention_mask = [b["last_speaker_sentences_attention_mask"] for b in batch]
    last_speaker_sentences_token_type_ids = [b["last_speaker_sentences_token_type_ids"] for b in batch]
    last_speaker_sentences_labels = [b["last_speaker_sentences_labels"] for b in batch]
    last_speaker_sentences_count = [b["last_speaker_sentences_count"] for b in batch]
    doc_ids = [b["doc_id"] for b in batch]
    target_input_ids = [b["target_input_ids"] for b in batch]
    target_attention_mask = [b["target_attention_mask"] for b in batch]
    target_token_type_ids = [b["target_token_type_ids"] for b in batch]

    dialogue_length = [len(sample) for sample in input_ids]
    dia_idx = []
    st = 0
    for num in dialogue_length:
        dia_idx.append([st, st + num])
        st += num

    user_history_length = [len(sample) for sample in last_speaker_sentences_input_ids]
    user_history_idx = []
    st = 0
    for num in user_history_length:
        user_history_idx.append([st, st + num])
        st += num

    max_lens = max((len(w) for sublist in input_ids for w in sublist), default=1)
    max_target_len = max((len(w) for w in target_input_ids), default=1)
    max_last_speaker_sentences_len = max((len(w) for sublist in last_speaker_sentences_input_ids for w in sublist), default=1)

    input_ids_flat = [_pad_sequence(seq, max_lens, pad_value=0) for sample in input_ids for seq in sample]
    attention_mask_flat = [_pad_sequence(seq, max_lens, pad_value=0) for sample in attention_mask for seq in sample]
    token_type_ids_flat = [_pad_sequence(seq, max_lens, pad_value=0) for sample in token_type_ids for seq in sample]

    last_speaker_sentences_input_ids_flat = [
        _pad_sequence(seq, max_last_speaker_sentences_len, pad_value=0)
        for sample in last_speaker_sentences_input_ids
        for seq in sample
    ]
    last_speaker_sentences_attention_mask_flat = [
        _pad_sequence(seq, max_last_speaker_sentences_len, pad_value=0)
        for sample in last_speaker_sentences_attention_mask
        for seq in sample
    ]
    last_speaker_sentences_token_type_ids_flat = [
        _pad_sequence(seq, max_last_speaker_sentences_len, pad_value=0)
        for sample in last_speaker_sentences_token_type_ids
        for seq in sample
    ]

    target_input_ids_pad = [_pad_sequence(seq, max_target_len, pad_value=0) for seq in target_input_ids]
    target_attention_mask_pad = [_pad_sequence(seq, max_target_len, pad_value=0) for seq in target_attention_mask]
    target_token_type_ids_pad = [_pad_sequence(seq, max_target_len, pad_value=0) for seq in target_token_type_ids]

    if len(input_ids_flat) == 0:
        input_ids_tensor = torch.zeros((0, max_lens), dtype=torch.long)
        attention_mask_tensor = torch.zeros((0, max_lens), dtype=torch.long)
        token_type_ids_tensor = torch.zeros((0, max_lens), dtype=torch.long)
    else:
        input_ids_tensor = torch.tensor(input_ids_flat, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_mask_flat, dtype=torch.long)
        token_type_ids_tensor = torch.tensor(token_type_ids_flat, dtype=torch.long)

    if len(last_speaker_sentences_input_ids_flat) == 0:
        last_speaker_sentences_input_ids_tensor = torch.zeros((0, max_last_speaker_sentences_len), dtype=torch.long)
        last_speaker_sentences_attention_mask_tensor = torch.zeros((0, max_last_speaker_sentences_len), dtype=torch.long)
        last_speaker_sentences_token_type_ids_tensor = torch.zeros((0, max_last_speaker_sentences_len), dtype=torch.long)
    else:
        last_speaker_sentences_input_ids_tensor = torch.tensor(last_speaker_sentences_input_ids_flat, dtype=torch.long)
        last_speaker_sentences_attention_mask_tensor = torch.tensor(last_speaker_sentences_attention_mask_flat, dtype=torch.long)
        last_speaker_sentences_token_type_ids_tensor = torch.tensor(last_speaker_sentences_token_type_ids_flat, dtype=torch.long)

    return {
        "input_ids": input_ids_tensor,
        "attention_mask": attention_mask_tensor,
        "token_type_ids": token_type_ids_tensor,
        "speakers": speakers,
        "label": torch.tensor(labels, dtype=torch.long),
        "all_labels": all_labels,
        "dia_idx": dia_idx,
        "user_history_idx": user_history_idx,
        "target": targets,
        "target_type": target_types,
        "last_speaker_sentences_input_ids": last_speaker_sentences_input_ids_tensor,
        "last_speaker_sentences_attention_mask": last_speaker_sentences_attention_mask_tensor,
        "last_speaker_sentences_token_type_ids": last_speaker_sentences_token_type_ids_tensor,
        "last_speaker_sentences_labels": last_speaker_sentences_labels,
        "last_speaker_sentences_count": torch.tensor(last_speaker_sentences_count, dtype=torch.long),
        "target_input_ids": torch.tensor(target_input_ids_pad, dtype=torch.long),
        "target_attention_mask": torch.tensor(target_attention_mask_pad, dtype=torch.long),
        "target_token_type_ids": torch.tensor(target_token_type_ids_pad, dtype=torch.long),
        "doc_id": doc_ids,
    }
