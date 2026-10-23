# !/usr/bin/python
# -*- coding: utf-8 -*-
# @author: xiongjie <xiongjiezk@163.com>
# @date: 2026/5/15

import json
import os
import sys
import time
import logging
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from transformers import (
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    TrainerCallback,
    EarlyStoppingCallback,
    set_seed as training_set_seed,
)
from transformers.utils import logging as hf_logging
from yaml import safe_load

from datasets.dataset_csd_2_qwen_1 import DialogueStanceDataset, dialogue_collate_fn
from models.model_sitpcl_12_qwen_1 import SITCL


hf_logging.set_verbosity_info()
hf_logging.enable_default_handler()
hf_logging.enable_explicit_format()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logging.getLogger("transformers.trainer").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)


@dataclass
class SITCLConfig:
    train_path: str = field(default="data/train_data.json")
    dev_path: str = field(default="data/dev_data.json")
    test_path: str = field(default="data/test_data.json")

    train_embedding_path: str = ""
    dev_embedding_path: str = ""
    test_embedding_path: str = ""
    sentence_csv: str = ""
    embedding_dim: int = 0
    hidden_dims: list[int] = field(default_factory=lambda: [1024, 512])
    activation: str = "relu"
    use_layernorm: bool = True
    enable_dual_pairs: bool = True
    dual_ce_weight: float = 1.0
    dual_share_user_history: bool = True
    use_history: bool = True
    use_causal_history: bool = True
    user_history_mode: str = "prefix"
    preload_embeddings: bool = True
    num_workers: int = 0
    prefetch_factor: int = 2
    resume_from_checkpoint: str = ""
    legacy_181700: bool = False
    training_backend: str = "trainer"
    eval_batchsize: int = 512
    match_legacy_eval_rng: bool = False  # Only used by the optional distmlp diagnostic backend.
    fp16: bool = False
    mlp_checkpoint: str = ""
    freeze_base_classifier: bool = False
    base_ce_weight: float = 0.0
    history_max_ratio: float = 0.0
    lr_scheduler_type: str = "linear"

    model_type: str = field(default="PPED_Qwen_Dual")
    result_root: str = field(default="./result")

    epoch_size: int = field(default=20)
    batchsize: int = field(default=16)
    patience: int = field(default=8)
    max_grad_norm: float = field(default=1.0)
    gradient_accumulation_steps: int = field(default=1)
    adam_epsilon: float = field(default=1e-8)
    warmup_proportion: float = field(default=0.1)

    other_lr: float = field(default=2e-5)
    weight_decay: float = field(default=1e-5)
    hidden_size: int = field(default=768)
    num_classes: int = field(default=3)
    num_heads: int = field(default=4)
    alpha: float = field(default=1.0)
    tau: float = field(default=0.1)
    lambda_distill: float = field(default=1.0)
    posterior_ce_weight: float = field(default=1.0)
    selector_tau: float = field(default=0.2)
    dropout: float = field(default=0.1)
    cuda_index: int = field(default=0)
    seed: int = field(default=1234)
    debug: bool = field(default=False)
    time: str = field(default="")
    model_type: str = field(default="PPED_Qwen_Dual")
    result_root: str = field(default="./result")

    def to_json_string(self, *args, **kwargs):
        cfg_dict = asdict(self)
        if "device" in cfg_dict:
            cfg_dict["device"] = str(cfg_dict["device"])
        return json.dumps(cfg_dict, indent=2, sort_keys=True, ensure_ascii=False)


def mtcsd_metrics(labels, preds):
    """Keep none examples; restrict F1 labels to favor/against, as train.py does."""
    labels, preds = np.asarray(labels), np.asarray(preds)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, labels=[0, 1], average="macro", zero_division=0)),
        "f1_micro": float(f1_score(labels, preds, labels=[0, 1], average="micro", zero_division=0)),
        "f1_favor": float(f1_score(labels, preds, labels=[0], average="macro", zero_division=0)),
        "f1_against": float(f1_score(labels, preds, labels=[1], average="macro", zero_division=0)),
        "f1_macro_3": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_micro_3": float(f1_score(labels, preds, average="micro", zero_division=0)),
    }


def compute_metrics_sitpcl(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    return mtcsd_metrics(np.asarray(labels).reshape(-1), np.argmax(logits, axis=-1))


class SITCLTrainer(Trainer):
    def get_train_dataloader(self):
        if self.args.world_size > 1:
            # Let Accelerate shard batches exactly once and advance sampler epochs.
            # Eval/predict args remain at zero workers to avoid the earlier fork OOM.
            model = self.model.module if hasattr(self.model, "module") else self.model
            names = ("dataloader_num_workers", "dataloader_persistent_workers", "dataloader_prefetch_factor")
            previous = {name: getattr(self.args, name) for name in names}
            workers = model.config.num_workers
            try:
                self.args.dataloader_num_workers = workers
                self.args.dataloader_persistent_workers = workers > 0
                self.args.dataloader_prefetch_factor = model.config.prefetch_factor if workers else None
                return super().get_train_dataloader()
            finally:
                for name, value in previous.items():
                    setattr(self.args, name, value)
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if data_collator is None:
            raise ValueError("Trainer: training requires a data_collator.")

        if self.args.world_size > 1:
            sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                shuffle=True,
                seed=self.args.data_seed,
                drop_last=self.args.dataloader_drop_last,
            )
        else:
            sampler = RandomSampler(train_dataset)

        if not getattr(self, "_train_dataloader_logged", False):
            logging.info(
                "[train-loader] world_size=%s | process_index=%s | local_rank=%s | sampler=%s | shuffle_each_epoch=%s | data_seed=%s",
                self.args.world_size,
                self.args.process_index,
                getattr(self.args, "local_rank", None),
                sampler.__class__.__name__,
                isinstance(sampler, DistributedSampler) or isinstance(sampler, RandomSampler),
                self.args.data_seed,
            )
            self._train_dataloader_logged = True

        model = self.model.module if hasattr(self.model, "module") else self.model
        workers = model.config.num_workers  # Training only; eval/predict always use 0.
        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=data_collator,
            num_workers=workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
            persistent_workers=workers > 0,
            **({"prefetch_factor": model.config.prefetch_factor} if workers > 0 else {}),
        )

    # Evaluation and prediction use Trainer's native loaders. Accelerate handles
    # rank sharding, gathered predictions and removal of padded final examples.

    def create_optimizer(self):
        if self.optimizer is None:
            model = self.model.module if hasattr(self.model, "module") else self.model
            decay, no_decay = [], []
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    # Match strucdist/main_dist.py, including its Sequential LayerNorm naming.
                    excluded = (parameter.ndim < 2 or name.endswith("bias")) if getattr(model.config, "legacy_181700", False) else any(term in name for term in ("bias", "LayerNorm.weight", "norm.weight", "norm.bias"))
                    (no_decay if excluded else decay).append(parameter)
            self.optimizer = AdamW(
                [{"params": decay, "weight_decay": model.config.weight_decay},
                 {"params": no_decay, "weight_decay": 0.0}],
                lr=model.config.other_lr, eps=model.config.adam_epsilon)
        return self.optimizer

    def create_scheduler(self, num_training_steps, optimizer=None):
        if self.lr_scheduler is None:
            if self.model.config.lr_scheduler_type == "linear":
                warmup = max(1, min(int(num_training_steps * self.model.config.warmup_proportion), max(1, num_training_steps)))
                def multiplier(step):
                    if step < warmup:
                        return float(step) / warmup
                    return max(0., 1. - float(step-warmup) / max(1, num_training_steps-warmup))
                self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer or self.optimizer, multiplier)
            else:
                return super().create_scheduler(num_training_steps, optimizer)
        return self.lr_scheduler

    def log(self, logs, start_time=None):
        model = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(model, "pop_debug_cache"):
            dbg = model.pop_debug_cache()
            logs = dict(logs)
            logs.update({f"dbg_{k}": v for k, v in dbg.items()})
        super().log(logs, start_time)
        if logs:
            keep_keys = ["loss", "grad_norm", "learning_rate", "epoch", "eval_loss", "eval_f1_macro", "eval_f1_avg", "eval_accuracy"]
            compact = []
            for key in keep_keys:
                if key in logs:
                    value = logs[key]
                    if isinstance(value, (int, float, np.floating)):
                        compact.append(f"{key}={float(value):.6f}")
                    else:
                        compact.append(f"{key}={value}")
            dbg_keys = [
                "dbg_dual_ce_loss",
                "dbg_dual_pair_fraction",
                "dbg_dual_user_aligned_fraction",
                "dbg_history_ratio",
                "dbg_base_ce_loss",
                "dbg_final_fused_cosine",
                "dbg_dialog_null_prob",
                "dbg_user_null_prob",
                "dbg_loss",
                "dbg_ce_loss",
                "dbg_distill_loss",
                "dbg_target_contrastive_loss",
                "dbg_posterior_ce_loss",
                "dbg_main_confidence",
                "dbg_main_entropy",
                "dbg_posterior_count",
                "dbg_history_len",
                "dbg_dialog_prior_prob_max",
                "dbg_user_prior_prob_max",
                "dbg_dialog_posterior_prob_max",
                "dbg_user_posterior_prob_max",
                "dbg_prior_pool_norm",
                "dbg_posterior_pool_norm",
                "dbg_target_norm",
            ]
            dbg_compact = []
            for key in dbg_keys:
                if key in logs:
                    value = logs[key]
                    if isinstance(value, (int, float, np.floating)):
                        dbg_compact.append(f"{key}={float(value):.6f}")
                    else:
                        dbg_compact.append(f"{key}={value}")
            if compact or dbg_compact:
                logging.info("[trainer] " + " | ".join(compact + dbg_compact))
    
    def training_step(self, model, inputs, *args, **kwargs):
        module = model.module if hasattr(model, "module") else model
        if hasattr(module, "set_step"):
            module.set_step(
                self.state.global_step,
                self.state.max_steps,
            )
        return super().training_step(model, inputs, *args, **kwargs)

class TestPredictionCallback(TrainerCallback):
    def __init__(self, trainer, eval_dataset, test_dataset, pred_dir):
        self.trainer = trainer
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.pred_dir = pred_dir

    @staticmethod
    def _is_main_process(args) -> bool:
        return getattr(args, "process_index", 0) == 0

    def _evaluate_dataset(self, dataset, metric_key_prefix="test", save_suffix="final"):
        predictions = self.trainer.predict(dataset)
        logits = predictions.predictions
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        preds = np.argmax(logits, axis=-1)
        trues = predictions.label_ids.reshape(-1)
        if len(preds) != len(dataset.data):
            raise ValueError("Prediction count does not match dataset rows")
        df = pd.DataFrame({
            "doc_id": [ex["doc_id"] for ex in dataset.data],
            "target": [ex["target"] for ex in dataset.data],
            "target_type": [ex.get("target_type", "") for ex in dataset.data],
            "true": trues, "pred": preds,
        })
        overall = mtcsd_metrics(trues, preds)
        rows = [{"target": target, "count": len(group),
                 **mtcsd_metrics(group["true"], group["pred"])}
                for target, group in df.groupby("target", sort=True)]
        per_target = pd.DataFrame(rows)
        metrics = {f"{metric_key_prefix}_{k}": v for k, v in overall.items()}
        if self._is_main_process(self.trainer.args):
            os.makedirs(self.pred_dir, exist_ok=True)
            stem = os.path.join(self.pred_dir, f"{save_suffix}_{metric_key_prefix}")
            df.to_csv(stem + ".csv", index=False)
            per_target.to_csv(stem + "_per_target.csv", index=False)
            with open(stem + "_metrics.json", "w", encoding="utf-8") as f:
                json.dump({"overall": overall, "per_target": rows}, f, ensure_ascii=False, indent=2)
            logging.info("%s overall: %s", metric_key_prefix.upper(), overall)
            logging.info("%s per-target metrics:\n%s", metric_key_prefix.upper(), per_target.to_string(index=False))
        return metrics

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_tag = "unknown"
        if state.epoch is not None:
            epoch_tag = f"epoch_{int(round(state.epoch))}"
        if self._is_main_process(args):
            logging.info(f"Epoch {state.epoch:.0f} finished, evaluating dev/test sets...")
        model = self.trainer.model.module if hasattr(self.trainer.model, "module") else self.trainer.model
        if hasattr(model, "reset_debug_cache"):
            model.reset_debug_cache()
        self._evaluate_dataset(self.eval_dataset, metric_key_prefix="dev", save_suffix=epoch_tag)
        self._evaluate_dataset(self.test_dataset, metric_key_prefix="test", save_suffix=epoch_tag)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self._is_main_process(args):
            logging.info("Training finished, running final dev/test prediction with best model...")
        model = self.trainer.model.module if hasattr(self.trainer.model, "module") else self.trainer.model
        if hasattr(model, "reset_debug_cache"):
            model.reset_debug_cache()
        self._evaluate_dataset(self.eval_dataset, metric_key_prefix="dev", save_suffix="final")
        self._evaluate_dataset(self.test_dataset, metric_key_prefix="test", save_suffix="final")


def main():
    start_time = time.time()

    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        cfg_path = sys.argv[1]
        logging.info(f"load config file: {cfg_path}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_dict = safe_load(f)
        if "user_history" in cfg_dict:
            raise ValueError("Unknown user_history switch. Use use_history: false to disable ALL history.")
        cfg = SITCLConfig(**cfg_dict)
    else:
        parser = HfArgumentParser(SITCLConfig)
        cfg = parser.parse_args_into_dataclasses()[0]

    # Resolve portable MT-CSD paths relative to this entry, independent of cwd.
    entry_dir = os.path.dirname(os.path.abspath(__file__))
    for name in ("train_path", "dev_path", "test_path", "train_embedding_path",
                 "dev_embedding_path", "test_embedding_path", "sentence_csv", "result_root"):
        value = getattr(cfg, name)
        if value and not os.path.isabs(value):
            setattr(cfg, name, os.path.abspath(os.path.join(entry_dir, value)))

    if cfg.training_backend != "trainer":
        raise ValueError("Dual-target entry requires training_backend: trainer")
    if cfg.legacy_181700 or cfg.mlp_checkpoint:
        raise ValueError("Dual-target entry uses standard Trainer; legacy/replay modes are unsupported")
    if cfg.dual_ce_weight < 0:
        raise ValueError("dual_ce_weight must be nonnegative")

    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        # All ranks must agree on output/checkpoint paths, even across a second boundary.
        from accelerate import PartialState
        distributed_state = PartialState()
        run_time = [(cfg.time or time.strftime("%Y%m%d-%H%M%S")) if distributed_state.is_main_process else None]
        torch.distributed.broadcast_object_list(run_time, src=0)
        cfg.time = run_time[0]
    elif not cfg.time:
        cfg.time = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{cfg.model_type}_{cfg.time}" if cfg.time else cfg.model_type
    base_dir = os.path.join(cfg.result_root, run_name)
    log_dir = os.path.join(base_dir, "log")
    pred_dir = os.path.join(base_dir, "pred")
    save_dir = os.path.join(base_dir, "save")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    pred_file = os.path.join(pred_dir, f"{cfg.seed}.csv")

    rank = int(os.environ.get("RANK", "0"))
    log_name = f"{cfg.seed}.log" if rank == 0 else f"{cfg.seed}.rank{rank}.log"
    file_handler = logging.FileHandler(os.path.join(log_dir, log_name), mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S"))
    logging.getLogger().addHandler(file_handler)

    logging.info(f"load SITCLConfig done, cost {time.time() - start_time:.3f}s, args list: \n{cfg}")
    logging.info(
        "[trial-config] "
        f"alpha={getattr(cfg, 'alpha', 'NA')} | "
        f"tau={getattr(cfg, 'tau', 'NA')} | "
        f"lambda_distill={getattr(cfg, 'lambda_distill', 'NA')} | "
        f"posterior_ce_weight={getattr(cfg, 'posterior_ce_weight', 'NA')} | "
        f"selector_tau={getattr(cfg, 'selector_tau', 'NA')} | "
        f"other_lr={getattr(cfg, 'other_lr', 'NA')} | "
        f"weight_decay={getattr(cfg, 'weight_decay', 'NA')} "
    )
    start_time = time.time()

    # Seed model initialization; Trainer performs its normal seed reset afterwards.
    if cfg.legacy_181700:
        from common import set_seed as original_seed
        original_seed(cfg.seed)
    else:
        training_set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.device = device

    train_set = DialogueStanceDataset(cfg.train_path, cfg.train_embedding_path, debug=cfg.debug, seed=cfg.seed, sentence_csv=cfg.sentence_csv, user_history_mode=cfg.user_history_mode, preload_embeddings=cfg.preload_embeddings, include_history=cfg.use_history, enable_pairs=cfg.enable_dual_pairs and cfg.dual_ce_weight > 0)
    dev_set = DialogueStanceDataset(cfg.dev_path, cfg.dev_embedding_path, debug=cfg.debug, seed=cfg.seed, sentence_csv=cfg.sentence_csv, user_history_mode=cfg.user_history_mode, preload_embeddings=cfg.preload_embeddings, include_history=cfg.use_history, enable_pairs=False)
    test_set = DialogueStanceDataset(cfg.test_path, cfg.test_embedding_path, debug=cfg.debug, seed=cfg.seed, sentence_csv=cfg.sentence_csv, user_history_mode=cfg.user_history_mode, preload_embeddings=cfg.preload_embeddings, include_history=cfg.use_history, enable_pairs=False)
    logging.info("Dual supervision: enabled=%s | weight=%s | share_aligned_user=%s | stats=%s",
                 train_set.enable_pairs, cfg.dual_ce_weight, cfg.dual_share_user_history, train_set.pair_stats)
    if train_set.enable_pairs and not train_set.pair_stats.get('paired', 0):
        logging.warning("No usable target pairs found; only original PPED losses will be trained.")
    logging.info(f"load dataset done, cost {time.time() - start_time:.3f}s")
    start_time = time.time()

    dims = {train_set.embedding_dim, dev_set.embedding_dim, test_set.embedding_dim}
    if len(dims) != 1 or cfg.embedding_dim not in (0, train_set.embedding_dim):
        raise ValueError("Configured/train/dev/test embedding dimensions disagree")
    cfg.embedding_dim = train_set.embedding_dim
    model = SITCL(cfg)
    logging.info(f"load model done, cost {time.time() - start_time:.3f}s")
    start_time = time.time()

    training_args = TrainingArguments(
        seed=cfg.seed,
        data_seed=cfg.seed,
        full_determinism=cfg.legacy_181700,
        output_dir=save_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        learning_rate=cfg.other_lr,
        per_device_train_batch_size=cfg.batchsize,
        per_device_eval_batch_size=cfg.eval_batchsize,
        num_train_epochs=cfg.epoch_size,
        logging_dir=os.path.join(log_dir, "tensorboard"),
        logging_strategy="steps",
        log_level="info",
        disable_tqdm=True,
        logging_steps=50,
        report_to="tensorboard",
        fp16=cfg.fp16 and torch.cuda.is_available(),
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_safetensors=False,
        max_grad_norm=cfg.max_grad_norm,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_proportion,
        lr_scheduler_type=cfg.lr_scheduler_type,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        # Base Trainer creates eval/test loaders. Never fork new workers after
        # training has grown the process memory footprint; training has its own setting.
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=False,
        remove_unused_columns=False,
        label_names=["label"],
        ddp_find_unused_parameters=True,
    )

    trainer = SITCLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=dev_set,
        compute_metrics=compute_metrics_sitpcl,
        data_collator=dialogue_collate_fn,
    )

    trainer.model_accepts_loss_kwargs = False
    if cfg.legacy_181700:
        logging.info("Historical mode=181700 reconstruction; teacher values=text+labels; no early stopping")
    logging.info("RNG policy=standard Trainer seed reset; no post-init RNG restore or legacy eval RNG tick")
    logging.info("Backend=trainer; use_history=%s; initialization=%s; eval_batchsize=%d",
                 cfg.use_history, "checkpoint" if cfg.mlp_checkpoint else "random/from scratch", cfg.eval_batchsize)
    logging.info("DataLoader workers: train=%d, dev/test=0; preload_embeddings=%s",
                 cfg.num_workers, cfg.preload_embeddings)

    if cfg.patience > 0 and not cfg.legacy_181700:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=cfg.patience))
    trainer.add_callback(TestPredictionCallback(trainer, dev_set, test_set, pred_dir))

    logging.info(f"trainer start, cost {time.time() - start_time:.3f}s")
    logging.info(
        "[train-start] world_size=%s | process_index=%s | local_rank=%s | train_sampler=%s | sampler_shuffle_each_epoch=%s",
        training_args.world_size,
        training_args.process_index,
        getattr(training_args, "local_rank", None),
        "Trainer/Accelerate distributed sharding" if training_args.world_size > 1 else "RandomSampler",
        True,
    )
    if cfg.resume_from_checkpoint:
        checkpoint = os.path.abspath(cfg.resume_from_checkpoint)
        if not os.path.isfile(os.path.join(checkpoint, "trainer_state.json")):
            raise ValueError(f"Not a Trainer checkpoint directory: {checkpoint}")
        logging.info("Resuming training from %s", checkpoint)
    else:
        checkpoint = None
    trainer.train(resume_from_checkpoint=checkpoint)

    # best_model_path = os.path.join(save_dir, "best_model")
    # trainer.save_model(best_model_path)
    # logging.info(f"Best model saved to {best_model_path}")

    logging.info("Training finished. Best checkpoint selected by dev favor/against macro-F1; final dev/test exported.")


if __name__ == "__main__":
    main()
