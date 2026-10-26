"""Dataclass + YAML configuration, following main_12_qwen_1.py."""
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from yaml import safe_load

DEFAULT_CONFIG = Path(__file__).resolve().parent / 'config/config_glan_qwen.yaml'


@dataclass
class GLANConfig:
    train_path: str = ''
    dev_path: str = ''
    test_path: str = ''
    train_embedding_path: str = ''
    dev_embedding_path: str = ''
    test_embedding_path: str = ''
    train_ids_path: str = ''
    dev_ids_path: str = ''
    test_ids_path: str = ''
    sentence_csv: str = ''
    target_store: str = ''
    result_root: str = '../result'
    model_type: str = 'GLAN_Qwen'
    time: str = ''
    hidden_size: int = 256
    hop: int = 3
    lambdaa: float = 0.2
    dropout: float = 0.2
    other_lr: float = 3e-4
    weight_decay: float = 1e-4
    batchsize: int = 64
    eval_batchsize: int = 64
    epoch_size: int = 30
    patience: int = 10
    max_grad_norm: float = 1.0
    seed: int = 123
    device: str = 'auto'
    cuda_index: int = 0
    model_dir: str = '/home/xiongjie/data/models/qwen3-embedding-8b'
    embedding_batchsize: int = 8
    embedding_max_length: int = 512
    embedding_dtype: str = 'bf16'
    embedding_device_map: str = 'none'
    embedding_normalize: bool = True
    trust_remote_code: bool = True

    def to_json_string(self):
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


def load_config(path=DEFAULT_CONFIG):
    path = Path(path).resolve()
    with path.open(encoding='utf-8-sig') as f:
        values = safe_load(f)
    if not isinstance(values, dict):
        raise ValueError('Configuration must be a YAML mapping')
    unknown = set(values) - {f.name for f in fields(GLANConfig)}
    if unknown:
        raise ValueError(f'Unknown configuration fields: {sorted(unknown)}')
    cfg = GLANConfig(**values)
    required = ('train_path', 'dev_path', 'test_path', 'train_embedding_path',
                'dev_embedding_path', 'test_embedding_path', 'target_store')
    for key in required:
        if not isinstance(getattr(cfg, key), str) or not getattr(cfg, key).strip():
            raise ValueError(f'Missing configuration field: {key}')
    # Independent of the shell working directory; applies to every configured path.
    for key in (*required, 'train_ids_path', 'dev_ids_path', 'test_ids_path',
                'sentence_csv', 'result_root', 'model_dir'):
        value = getattr(cfg, key)
        if value:
            resolved = Path(value).expanduser()
            if not resolved.is_absolute():
                resolved = path.parent / resolved
            setattr(cfg, key, str(resolved.resolve()))
    for key in ('hidden_size', 'hop', 'batchsize', 'eval_batchsize', 'epoch_size', 'patience',
                'embedding_batchsize', 'embedding_max_length'):
        if type(getattr(cfg, key)) is not int or getattr(cfg, key) < 1:
            raise ValueError(f'{key} must be a positive integer')
    if cfg.embedding_dtype not in ('auto', 'bf16', 'fp16', 'fp32'):
        raise ValueError('Invalid embedding_dtype')
    if cfg.embedding_device_map not in ('none', 'auto'):
        raise ValueError('embedding_device_map must be none or auto')
    for key in ('embedding_normalize', 'trust_remote_code'):
        if type(getattr(cfg, key)) is not bool:
            raise ValueError(f'{key} must be a YAML boolean')
    if cfg.other_lr <= 0 or cfg.weight_decay < 0 or cfg.max_grad_norm <= 0:
        raise ValueError('Invalid learning rate, weight decay, or max_grad_norm')
    if not 0 <= cfg.dropout < 1:
        raise ValueError('dropout must be in [0,1)')
    if not cfg.result_root or not cfg.model_type:
        raise ValueError('result_root and model_type must not be empty')
    for key in ('model_type', 'time'):
        if any(c in getattr(cfg, key) for c in '/\\:') or getattr(cfg, key) in ('.', '..'):
            raise ValueError(f'{key} must be a simple name, not a path')
    return cfg
