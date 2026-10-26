import logging
import sys
from datetime import datetime
import csv
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import GLANDataset, collate
from model import GLANQwen
from metrics import scores, metrics_by_scope
from configuration import DEFAULT_CONFIG, load_config


def run(model, loader, device, optimizer=None, prefix='eval', max_grad_norm=1.0):
    model.train(optimizer is not None)
    labels, preds, rows, total_loss = [], [], [], 0.
    with torch.set_grad_enabled(optimizer is not None):
        for b in loader:
            y = b['labels'].to(device)
            logits = model(b['utterances'].to(device), b['spans'], b['target_embeddings'].to(device))
            loss = torch.nn.functional.cross_entropy(logits, y)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            predicted = logits.detach().argmax(-1).cpu().tolist()
            labels.extend(b['labels'].tolist())
            preds.extend(predicted)
            rows.extend(dict(**meta, label=label, prediction=pred)
                        for meta, label, pred in zip(b['metadata'], b['labels'].tolist(), predicted))
            total_loss += loss.item() * len(y)
    if not labels:
        raise ValueError('Empty split')
    result = scores(labels, preds)
    result['loss'] = total_loss / len(labels)
    result.update(metrics_by_scope(labels, preds, [r['target_type'] for r in rows], prefix))
    return result, rows


def main(config_path=DEFAULT_CONFIG):
    args = load_config(config_path)
    if args.device == 'auto':
        args.device = f'cuda:{args.cuda_index}' if torch.cuda.is_available() else 'cpu'
    if not args.time:
        args.time = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    datasets = {s: GLANDataset(getattr(args, name+'_path'), getattr(args, name+'_embedding_path'),
                args.target_store, args.sentence_csv, getattr(args, name+'_ids_path'))
                for s, name in (('train', 'train'), ('valid', 'dev'), ('test', 'test'))}
    target_sets = {s: {r['target'] for r in ds.base.data} for s, ds in datasets.items()}
    for a, b in (('train', 'valid'), ('train', 'test'), ('valid', 'test')):
        if target_sets[a] & target_sets[b]:
            raise ValueError(f'ZS-CSD requires disjoint targets: {a}/{b} overlap')
    if len({ds.embedding_dim for ds in datasets.values()}) != 1:
        raise ValueError('Embedding dimensions differ between splits')
    out = Path(args.result_root) / f'{args.model_type}_{args.time}'
    out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(out/f'{args.seed}.log', encoding='utf-8')],
                        force=True)
    logging.info('Loaded config: %s\n%s', Path(config_path).resolve(), args.to_json_string())
    (out/'config.json').write_text(args.to_json_string(), encoding='utf-8')
    loaders = {s: DataLoader(ds, batch_size=args.batchsize if s == 'train' else args.eval_batchsize,
                            shuffle=s == 'train', collate_fn=collate)
               for s, ds in datasets.items()}
    model = GLANQwen(datasets['train'].embedding_dim, args.hidden_size, args.hop,
                     args.lambdaa, args.dropout).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.other_lr, weight_decay=args.weight_decay)
    best, stale, best_epoch = -1., 0, 0
    for epoch in range(1, args.epoch_size + 1):
        train, _ = run(model, loaders['train'], args.device, optimizer, prefix='train',
                       max_grad_norm=args.max_grad_norm)
        valid, _ = run(model, loaders['valid'], args.device, prefix='dev')
        log = dict(epoch=epoch, train=train, valid=valid)
        logging.info(json.dumps(log))
        with (out/'history.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(log) + '\n')
        if valid['f1_macro'] > best:
            best, stale, best_epoch = valid['f1_macro'], 0, epoch
            torch.save(model.state_dict(), out/'best.pt')
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(torch.load(out/'best.pt', map_location=args.device, weights_only=True))
    result = dict(best_epoch=best_epoch, best_valid_macro_f1=best,
                  label_map=dict(favor=0, against=1, neutral=2), metric_scale='0-1')
    for split, prefix in (('valid', 'dev'), ('test', 'test')):
        evaluated, rows = run(model, loaders[split], args.device, prefix=prefix)
        result.update({key: value for key, value in evaluated.items() if key.startswith(prefix + '_')})
        result[prefix + '_loss'] = evaluated['loss']
        for scope in ('mixed', 'c', 'n'):
            fields = ('f1_macro', 'f1_favor', 'f1_against', 'f1_neutral', 'f1_avg', 'accuracy')
            logging.info(f'{prefix.upper()} {scope} metrics | ' + ' | '.join(
                f'{key}={evaluated[f"{prefix}_{scope}_{key}"]:.6f}' for key in fields))
        with (out/f'final_{prefix}.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['doc_id', 'target_type', 'true', 'pred'])
            writer.writeheader()
            writer.writerows(dict(doc_id=r['doc_id'], target_type=r['target_type'],
                                  true=r['label'], pred=r['prediction']) for r in rows)
    (out/'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    with (out/'test_predictions.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logging.info('Final results:\n%s', json.dumps(result, indent=2))
    logging.info('Results saved to %s', out)


if __name__ == '__main__':
    if len(sys.argv) > 2 or (len(sys.argv) == 2 and Path(sys.argv[1]).suffix.lower() not in ('.yaml', '.yml')):
        raise SystemExit('Run train.py with no arguments, or supply one YAML configuration path.')
    main(sys.argv[1] if len(sys.argv) == 2 else DEFAULT_CONFIG)
