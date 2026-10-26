"""Encode target text only (no stance annotations); reuse the utterance encoder checkpoint."""
import sys
from types import SimpleNamespace
import json
from pathlib import Path
import numpy as np
import importlib.util
from configuration import DEFAULT_CONFIG, load_config


def main(config_path=DEFAULT_CONFIG):
    args = load_config(config_path)
    import torch
    import torch.nn.functional as F
    paths = [args.train_path, args.dev_path, args.test_path]
    targets = set()
    for path in paths:
        records = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        if not isinstance(records, list) or not records:
            raise ValueError(f'Expected a nonempty record list: {path}')
        for row, record in enumerate(records):
            target = record.get('target') if isinstance(record, dict) else None
            if not isinstance(target, str) or not target.strip():
                raise ValueError(f'Invalid target at {path}, row {row}')
            targets.add(target)
    targets = sorted(targets)
    output = Path(args.target_store)
    if output.suffix.lower() != '.npz':
        raise ValueError('target_store must end in .npz')
    source = Path(__file__).resolve().parents[1] / 'preprocess/build_embedding_qwen_target_final.py'
    spec = importlib.util.spec_from_file_location('_qwen_feature_builder', source)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    device = args.device
    if device == 'auto':
        device = f'cuda:{args.cuda_index}' if torch.cuda.is_available() else 'cpu'
    cfg = SimpleNamespace(model_dir=args.model_dir, device=device, dtype=args.embedding_dtype,
                          device_map=args.embedding_device_map, trust_remote_code=args.trust_remote_code)
    tokenizer, model, device = builder.load_model(cfg)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(targets), args.embedding_batchsize):
            batch = tokenizer(targets[start:start + args.embedding_batchsize], padding=True,
                              truncation=True, max_length=args.embedding_max_length, return_tensors='pt').to(device)
            vectors = model(**batch).last_hidden_state[:, -1].float()
            if args.embedding_normalize:
                vectors = F.normalize(vectors, p=2, dim=1)
            chunks.append(vectors.cpu().numpy())
    vectors = np.concatenate(chunks)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError('Encoder returned invalid/nonfinite target embeddings')
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, targets=np.asarray(targets), embeddings=vectors.astype(np.float32))
    output.with_suffix('.metadata.json').write_text(json.dumps(
        dict(model=args.model_dir, inputs='raw target text; no instruction', normalized=args.embedding_normalize,
             pooling='last token; left padding', max_length=args.embedding_max_length, dtype=args.embedding_dtype,
             device=str(device), device_map=args.embedding_device_map, batch_size=args.embedding_batchsize,
             sources=paths, target_count=len(targets), embedding_dim=vectors.shape[1],
             config_path=str(Path(config_path).resolve())), indent=2), encoding='utf-8')
    print(f'Saved {len(targets)} target embeddings {vectors.shape} to {output}', flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 2 or (len(sys.argv) == 2 and Path(sys.argv[1]).suffix.lower() not in ('.yaml', '.yml')):
        raise SystemExit('Run build_targets.py with no arguments, or supply one YAML configuration path.')
    main(sys.argv[1] if len(sys.argv) == 2 else DEFAULT_CONFIG)
