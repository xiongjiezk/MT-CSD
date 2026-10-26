import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import ast
import numpy as np
import torch
from data import GLANDataset, collate
from model import GLANQwen
from metrics import scores, metrics_by_scope
from sklearn.metrics import accuracy_score, f1_score
from configuration import load_config, DEFAULT_CONFIG


class GLANTests(unittest.TestCase):
    def test_incompatible_csv_diagnostic(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'sentence.csv'
            path.write_text('global_id,sentence\n0,example\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'actual columns=.*global_id.*sentence'):
                GLANDataset('unused.json', 'unused.npy', 'unused.npz', sentence_csv=str(path))

    def test_target_builder_uses_yaml(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        import build_targets
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            values = dict(model_dir='./model', target_store='targets.npz', device='cpu',
                          embedding_batchsize=1, embedding_max_length=23, embedding_dtype='fp32')
            for split in ('train', 'dev', 'test'):
                values[split+'_path'] = split+'.json'
                values[split+'_embedding_path'] = 'unused.npy'
                (root/(split+'.json')).write_text(json.dumps([{'target': split}]))
            config = root/'config.yaml'
            config.write_text(json.dumps(values))
            calls = []
            class Batch(dict):
                def to(self, device):
                    return self
            def tokenizer(texts, **kwargs):
                calls.append((texts, kwargs))
                return Batch(input_ids=torch.ones(len(texts), 2, dtype=torch.long))
            def model(input_ids):
                return SimpleNamespace(last_hidden_state=torch.ones(len(input_ids), 2, 4))
            def load_model(cfg):
                self.assertEqual(cfg.model_dir, str(root/'model'))
                self.assertEqual(cfg.dtype, 'fp32')
                self.assertEqual(cfg.device, 'cpu')
                return tokenizer, model, 'cpu'
            fake_module = SimpleNamespace(load_model=load_model)
            fake_spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda _: None))
            with patch.object(build_targets.importlib.util, 'spec_from_file_location', return_value=fake_spec), \
                 patch.object(build_targets.importlib.util, 'module_from_spec', return_value=fake_module):
                build_targets.main(config)
            with np.load(root/'targets.npz', allow_pickle=False) as store:
                self.assertEqual(store['targets'].tolist(), ['dev', 'test', 'train'])
                np.testing.assert_allclose(np.linalg.norm(store['embeddings'], axis=1), 1.)
            self.assertEqual(len(calls), 3)
            self.assertTrue(all(c[1]['max_length'] == 23 for c in calls))
            metadata = json.loads((root/'targets.metadata.json').read_text())
            self.assertEqual(metadata['batch_size'], 1)
            self.assertEqual(metadata['embedding_dim'], 4)

    def test_configuration(self):
        cfg = load_config()
        self.assertEqual(cfg.hidden_size, 256)
        self.assertEqual(Path(cfg.train_path).name, 'train_data.json')
        self.assertEqual(Path(cfg.result_root), DEFAULT_CONFIG.parent.parent / 'result')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'bad.yaml'
            path.write_text('unknown_setting: 1', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Unknown configuration'):
                load_config(path)
            path.write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Missing configuration'):
                load_config(path)

    def test_metrics_match_original_script(self):
        # Execute only the original metric functions, without loading its model,
        # Trainer, datasets, or any third-party training services.
        source = Path(__file__).resolve().parents[2] / 'csd/main_12_qwen_1.py'
        tree = ast.parse(source.read_text(encoding='utf-8-sig'))
        names = ('compute_metrics_sitpcl', '_metrics_by_scope')
        functions = [node for node in ast.walk(tree)
                     if isinstance(node, ast.FunctionDef) and node.name in names]
        self.assertEqual(len(functions), 2)
        for node in functions:
            node.decorator_list = []
        namespace = dict(np=np, f1_score=f1_score, accuracy_score=accuracy_score)
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
        cases = [
            ([0, 1, 2, 0, 1, 2], [0, 2, 2, 1, 1, 0], ['c']*3 + ['n']*3),
            ([0, 2, 2], [0, 0, 2], ['c']*3),  # Missing stance class and empty n scope.
            ([1], [1], ['n']),
            ([0, 1, 2], [2, 1, 0], ['', 'unknown', 'n']),
            ([], [], []),
        ]
        for labels, preds, types in cases:
            for prefix in ('dev', 'test'):
                self.assertEqual(metrics_by_scope(labels, preds, types, prefix),
                                 namespace['_metrics_by_scope'](labels, preds, types, prefix))
            if labels:
                logits = np.eye(3)[preds]
                self.assertEqual(scores(labels, preds),
                                 namespace['compute_metrics_sitpcl']((logits, np.asarray(labels))))

    def test_variable_lengths_batch_independence_and_gradients(self):
        torch.manual_seed(5)
        model = GLANQwen(12, hidden_dim=8, dropout=0).eval()
        x, targets = torch.randn(13, 12), torch.randn(2, 12)
        batched = model(x, [(0, 1), (1, 13)], targets)
        separate = torch.cat([model(x[:1], [(0, 1)], targets[:1]),
                              model(x[1:], [(0, 12)], targets[1:])])
        torch.testing.assert_close(batched, separate)
        changed = targets.clone()
        changed[0] += 10
        torch.testing.assert_close(batched[1], model(x, [(0, 1), (1, 13)], changed)[1])
        batched.square().sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))

    def test_refinement_matches_source(self):
        model = GLANQwen(8, 8, dropout=0)
        sentence, state = torch.randn(5, 8), torch.randn(5, 8)
        original = state.clone()
        mask = torch.tensor([0, 0, 0, 0, 1]).float()[:, None]
        for _ in range(model.hop - 1):
            alpha = (mask * original) @ sentence.T
            interaction = alpha.T[:, -1:] * sentence
            original = model.lambdaa * model.layer_norm1(torch.sigmoid(interaction)) + sentence
        torch.testing.assert_close(model.refine(state, sentence), original[-1])

    def test_data_and_full_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rng = np.random.default_rng(123)
            targets = ['train-target', 'valid-target', 'test-target']
            np.savez(root/'targets.npz', targets=np.asarray(targets), embeddings=rng.normal(size=(3, 12)))
            cfg = dict(target_store='targets.npz', result_root='.', model_type='run', time='smoke',
                       epoch_size=1, hidden_size=8, batchsize=3, eval_batchsize=2, device='cpu')
            command = [sys.executable, str(Path(__file__).with_name('train.py')), str(root/'config.yaml')]
            for split, target in zip(('train', 'valid', 'test'), targets):
                records = [dict(id=i, target=target, target_type='c' if i < 2 else 'n', sentences=[f'u{j}' for j in range(i+1)],
                                speakers=list(range(i+1)), label=i, all_labels=[0]*i+[i]) for i in range(3)]
                (root/f'{split}.json').write_text(json.dumps(records), encoding='utf-8')
                np.save(root/f'{split}.npy', rng.normal(size=(3, 12)))
                np.save(root/f'{split}_ids.npy', np.arange(3))
                name = 'dev' if split == 'valid' else split
                cfg[name+'_path'] = f'{split}.json'
                cfg[name+'_embedding_path'] = f'{split}.npy'
            (root/'config.yaml').write_text(json.dumps(cfg), encoding='utf-8')
            ds = GLANDataset(root/'train.json', root/'train.npy', root/'targets.npz')
            before = ds[2]['utterances'].clone()
            ds.base.data[2]['all_labels'] = [2, 1, 0]
            torch.testing.assert_close(before, ds[2]['utterances'])
            self.assertNotIn('all_labels', collate([ds[2]]))
            ds.base.matrix._mmap.close()
            del ds
            result = subprocess.run(command, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((root/'run_smoke/metrics.json').read_text())
            self.assertEqual(report['best_epoch'], 1)
            self.assertTrue((root/'run_smoke/test_predictions.csv').exists())
            self.assertTrue((root/'run_smoke/123.log').exists())
            import csv
            for prefix in ('dev', 'test'):
                with (root/f'run_smoke/final_{prefix}.csv').open(encoding='utf-8-sig', newline='') as f:
                    rows = list(csv.DictReader(f))
                expected = metrics_by_scope([int(r['true']) for r in rows],
                                            [int(r['pred']) for r in rows],
                                            [r['target_type'] for r in rows], prefix)
                self.assertEqual({k: report[k] for k in expected}, expected)
            # Incomplete split-final features must fail rather than fill missing history with zero.
            records = json.loads((root/'train.json').read_text())
            records[-1]['sentences'][0] = 'missing'
            (root/'train.json').write_text(json.dumps(records))
            with self.assertRaisesRegex(ValueError, 'Missing target'):
                GLANDataset(root/'train.json', root/'train.npy', root/'targets.npz')
            import gc
            gc.collect()


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
