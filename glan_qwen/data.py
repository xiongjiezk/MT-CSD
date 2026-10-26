"""Reuse PPED's exact target+sentence feature lookup; no historical label input."""
import importlib.util
import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

source = Path(__file__).resolve().parents[1] / 'pped/datasets/dataset_csd_2_qwen_1.py'
spec = importlib.util.spec_from_file_location('_pped_qwen_dataset', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class GLANDataset(Dataset):
    def __init__(self, json_path, embedding_path, target_store, sentence_csv='', ids_path=''):
        # Diagnose the cache format before loading the (potentially large) matrix.
        if sentence_csv:
            with open(sentence_csv, encoding='utf-8-sig', newline='') as f:
                columns = csv.DictReader(f).fieldnames or []
            if not {'global_id', 'target', 'sentence'}.issubset(columns):
                raise ValueError(
                    f'Incompatible sentence_csv: {sentence_csv}; actual columns={columns!r}. '
                    'Global features require global_id,target,sentence. To use the original '
                    'PPED split features, set sentence_csv: "" and set train/dev/test_embedding_path '
                    'to qwen3_embedding_8b_target_final/train.npy, dev.npy, test.npy '
                    '(with matching *_ids.npy). Do not merely add a target column to an '
                    'unverified embedding cache: that does not make its vectors target-conditioned.')
        self.base = module.DialogueStanceDataset(
            json_path, embedding_path, sentence_csv=sentence_csv, ids_path=ids_path,
            user_history_mode='prefix', preload_embeddings=False, enable_pairs=False)
        self.embedding_dim = self.base.embedding_dim
        with np.load(target_store, allow_pickle=False) as store:
            names = store['targets'].tolist()
            vectors = np.array(store['embeddings'], dtype=np.float32, copy=True)
        if (vectors.shape != (len(names), self.embedding_dim) or len(set(names)) != len(names)
                or not np.isfinite(vectors).all()):
            raise ValueError('Invalid target store: need unique targets and finite [T,D] embeddings')
        self.targets = dict(zip(names, vectors))
        missing = {r['target'] for r in self.base.data} - self.targets.keys()
        if missing:
            raise ValueError(f'Missing target embeddings: {sorted(missing)[:5]}')
        if any(r['label'] is None for r in self.base.data):
            raise ValueError('All records must have final stance labels')

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        r = self.base.data[i]
        vectors = np.array(self.base.matrix[r['gids']], dtype=np.float32, copy=True)
        if not np.isfinite(vectors).all():
            raise ValueError('Nonfinite utterance embeddings')
        return dict(utterances=torch.from_numpy(vectors),
                    target_embedding=torch.from_numpy(self.targets[r['target']].copy()),
                    label=r['label'], doc_id=r['doc_id'], target=r['target'], target_type=r['target_type'])


def collate(batch):
    lengths = [len(b['utterances']) for b in batch]
    ends = np.cumsum(lengths).tolist()
    return dict(utterances=torch.cat([b['utterances'] for b in batch]),
                spans=list(zip([0] + ends[:-1], ends)),
                target_embeddings=torch.stack([b['target_embedding'] for b in batch]),
                labels=torch.tensor([b['label'] for b in batch]),
                metadata=[{k: b[k] for k in ('doc_id', 'target', 'target_type')} for b in batch])
