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
)
from transformers.utils import logging as hf_logging
from yaml import safe_load

from datasets.dataset_csd_2 import DialogueStanceDataset, dialogue_collate_fn
from models.model_sitpcl_12 import SITCL
from common import set_seed


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

    bert_dir: str = field(default="./plm/chinese-roberta-wwm-ext/")
    model_type: str = field(default="SITCL_12")
    result_root: str = field(default="./result")

    epoch_size: int = field(default=20)
    batchsize: int = field(default=16)
    patience: int = field(default=10)
    max_grad_norm: float = field(default=1.0)
    gradient_accumulation_steps: int = field(default=1)
    adam_epsilon: float = field(default=1e-8)
    warmup_proportion: float = field(default=0.1)

    bert_lr: float = field(default=1e-5)
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
    model_type: str = field(default="SITCL_12")
    result_root: str = field(default="./result")

    def to_json_string(self, *args, **kwargs):
        cfg_dict = asdict(self)
        if "device" in cfg_dict:
            cfg_dict["device"] = str(cfg_dict["device"])
        return json.dumps(cfg_dict, indent=2, sort_keys=True, ensure_ascii=False)


def compute_metrics_sitpcl(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)

    f1_macro = f1_score(y_true=labels, y_pred=preds, average="macro")
    f1_per_class = f1_score(y_true=labels, y_pred=preds, average=None)
    if len(f1_per_class) >= 3:
        favor, against, neutral = f1_per_class[:3]
    else:
        favor = against = neutral = 0.0
    f1_avg = (favor + against) / 2.0
    acc = accuracy_score(y_true=labels, y_pred=preds)

    return {
        "f1_macro": f1_macro,
        "f1_favor": favor,
        "f1_against": against,
        "f1_neutral": neutral,
        "f1_avg": f1_avg,
        "accuracy": acc,
    }


class SITCLTrainer(Trainer):
    def get_train_dataloader(self):
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

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
            persistent_workers=self.args.dataloader_num_workers > 0,
        )

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        model = self.model.module if hasattr(self.model, "module") else self.model
        param_optimizer = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        no_decay = ["bias", "LayerNorm.weight"]

        bert_lr = float(getattr(model.config, "bert_lr", model.config.other_lr))
        other_lr = float(model.config.other_lr)
        weight_decay = float(model.config.weight_decay)
        adam_epsilon = float(model.config.adam_epsilon)

        def _is_bert_param(name: str) -> bool:
            return name.startswith("bert.") or name.startswith("utterance_proj.")

        bert_decay = [p for n, p in param_optimizer if _is_bert_param(n) and not any(nd in n for nd in no_decay)]
        bert_nodecay = [p for n, p in param_optimizer if _is_bert_param(n) and any(nd in n for nd in no_decay)]
        other_decay = [p for n, p in param_optimizer if not _is_bert_param(n) and not any(nd in n for nd in no_decay)]
        other_nodecay = [p for n, p in param_optimizer if not _is_bert_param(n) and any(nd in n for nd in no_decay)]

        optimizer_grouped_parameters = []
        if bert_decay:
            optimizer_grouped_parameters.append({"params": bert_decay, "weight_decay": weight_decay, "lr": bert_lr})
        if bert_nodecay:
            optimizer_grouped_parameters.append({"params": bert_nodecay, "weight_decay": 0.0, "lr": bert_lr})
        if other_decay:
            optimizer_grouped_parameters.append({"params": other_decay, "weight_decay": weight_decay, "lr": other_lr})
        if other_nodecay:
            optimizer_grouped_parameters.append({"params": other_nodecay, "weight_decay": 0.0, "lr": other_lr})

        self.optimizer = AdamW(optimizer_grouped_parameters, eps=adam_epsilon)

        logging.info(
            "[optimizer] differential lr policy: "
            f"bert_lr={bert_lr:.2e}, other_lr={other_lr:.2e}, weight_decay={weight_decay:.2e}, adam_epsilon={adam_epsilon:.2e}"
        )
        logging.info(
            "[optimizer] parameter counts: "
            f"bert={sum(p.numel() for p in bert_decay + bert_nodecay)}, "
            f"other={sum(p.numel() for p in other_decay + other_nodecay)}"
        )
        return self.optimizer

    def log(self, logs, start_time=None):
        model = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(model, "pop_debug_cache"):
            dbg = model.pop_debug_cache()
            logs = dict(logs)
            logs.update({f"dbg_{k}": v for k, v in dbg.items()})
        super().log(logs)
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
    
    def training_step(self, model, inputs):
        module = model.module if hasattr(model, "module") else model
        if hasattr(module, "set_step"):
            module.set_step(
                self.state.global_step,
                self.state.max_steps,
            )
        return super().training_step(model, inputs)

class TestPredictionCallback(TrainerCallback):
    def __init__(self, trainer, eval_dataset, test_dataset, pred_dir):
        self.trainer = trainer
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.pred_dir = pred_dir

    @staticmethod
    def _is_main_process(args) -> bool:
        return getattr(args, "process_index", 0) == 0 or getattr(args, "local_process_index", 0) == 0

    @staticmethod
    def _metrics_by_scope(y_true, y_pred, target_types, prefix):
        metrics = {}
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        target_types = np.asarray(target_types)

        scopes = {
            "mixed": np.ones_like(target_types, dtype=bool),
            "c": target_types == "c",
            "n": target_types == "n",
        }

        for scope_name, mask in scopes.items():
            if mask.sum() == 0:
                metrics.update({
                    f"{prefix}_{scope_name}_f1_macro": 0.0,
                    f"{prefix}_{scope_name}_f1_favor": 0.0,
                    f"{prefix}_{scope_name}_f1_against": 0.0,
                    f"{prefix}_{scope_name}_f1_neutral": 0.0,
                    f"{prefix}_{scope_name}_f1_avg": 0.0,
                    f"{prefix}_{scope_name}_accuracy": 0.0,
                })
                continue

            yt = y_true[mask]
            yp = y_pred[mask]
            f1_macro = f1_score(y_true=yt, y_pred=yp, average="macro")
            f1_per_class = f1_score(y_true=yt, y_pred=yp, average=None)
            if len(f1_per_class) >= 3:
                favor, against, neutral = f1_per_class[:3]
            else:
                favor = against = neutral = 0.0
            f1_avg = (favor + neutral + against) / 3.0
            acc = accuracy_score(y_true=yt, y_pred=yp)
            metrics.update({
                f"{prefix}_{scope_name}_f1_macro": f1_macro,
                f"{prefix}_{scope_name}_f1_favor": favor,
                f"{prefix}_{scope_name}_f1_against": against,
                f"{prefix}_{scope_name}_f1_neutral": neutral,
                f"{prefix}_{scope_name}_f1_avg": f1_avg,
                f"{prefix}_{scope_name}_accuracy": acc,
            })
        return metrics

    def _evaluate_dataset(self, dataset, metric_key_prefix="test", save_suffix="final"):
        predictions = self.trainer.predict(dataset)
        logits = predictions.predictions
        labels = predictions.label_ids

        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        preds = np.argmax(logits, axis=-1)
        trues = labels.reshape(-1)
        doc_ids = [ex["doc_id"] for ex in dataset.data]
        target_types = [ex.get("target_type", "") for ex in dataset.data]
        is_main = self._is_main_process(self.trainer.args)

        if is_main:
            result = {"doc_id": doc_ids, "target_type": target_types, "true": trues.tolist(), "pred": preds.tolist()}
            df = pd.DataFrame(result)
            os.makedirs(self.pred_dir, exist_ok=True)
            pred_path = os.path.join(self.pred_dir, f"{save_suffix}_{metric_key_prefix}.csv")
            df.to_csv(pred_path, index=False)

            metrics = self._metrics_by_scope(trues, preds, target_types, metric_key_prefix)
            logging.info(
                f"{metric_key_prefix.upper()} mixed metrics | "
                f"f1_macro={metrics[f'{metric_key_prefix}_mixed_f1_macro']:.6f} | "
                f"f1_favor={metrics[f'{metric_key_prefix}_mixed_f1_favor']:.6f} | "
                f"f1_against={metrics[f'{metric_key_prefix}_mixed_f1_against']:.6f} | "
                f"f1_neutral={metrics[f'{metric_key_prefix}_mixed_f1_neutral']:.6f} | "
                f"f1_avg={metrics[f'{metric_key_prefix}_mixed_f1_avg']:.6f} | "
                f"accuracy={metrics[f'{metric_key_prefix}_mixed_accuracy']:.6f}"
            )
            for scope_name in ["c", "n"]:
                logging.info(
                    f"{metric_key_prefix.upper()} {scope_name} metrics | "
                    f"f1_macro={metrics[f'{metric_key_prefix}_{scope_name}_f1_macro']:.6f} | "
                    f"f1_favor={metrics[f'{metric_key_prefix}_{scope_name}_f1_favor']:.6f} | "
                    f"f1_against={metrics[f'{metric_key_prefix}_{scope_name}_f1_against']:.6f} | "
                    f"f1_neutral={metrics[f'{metric_key_prefix}_{scope_name}_f1_neutral']:.6f} | "
                    f"f1_avg={metrics[f'{metric_key_prefix}_{scope_name}_f1_avg']:.6f} | "
                    f"accuracy={metrics[f'{metric_key_prefix}_{scope_name}_accuracy']:.6f}"
                )
            logging.info(f"{metric_key_prefix.upper()} predictions saved to {pred_path}")
        return self._metrics_by_scope(trues, preds, target_types, metric_key_prefix)

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
        cfg = SITCLConfig(**cfg_dict)
    else:
        parser = HfArgumentParser(SITCLConfig)
        cfg = parser.parse_args_into_dataclasses()[0]

    run_name = f"{cfg.model_type}_{cfg.time}" if cfg.time else cfg.model_type
    base_dir = os.path.join(cfg.result_root, run_name)
    log_dir = os.path.join(base_dir, "log")
    pred_dir = os.path.join(base_dir, "pred")
    save_dir = os.path.join(base_dir, "save")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    pred_file = os.path.join(pred_dir, f"{cfg.seed}.csv")

    file_handler = logging.FileHandler(os.path.join(log_dir, f"{cfg.seed}.log"), mode="w", encoding="utf-8")
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
        f"bert_lr={getattr(cfg, 'bert_lr', 'NA')} | "
        f"other_lr={getattr(cfg, 'other_lr', 'NA')} | "
        f"weight_decay={getattr(cfg, 'weight_decay', 'NA')} "
    )
    start_time = time.time()

    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.device = device

    train_set = DialogueStanceDataset(cfg.train_path, cfg.bert_dir, debug=cfg.debug, shuffle_data=False, seed=cfg.seed)
    dev_set = DialogueStanceDataset(cfg.dev_path, cfg.bert_dir, debug=cfg.debug, shuffle_data=False, seed=cfg.seed)
    test_set = DialogueStanceDataset(cfg.test_path, cfg.bert_dir, debug=cfg.debug, shuffle_data=False, seed=cfg.seed)
    logging.info(f"load dataset done, cost {time.time() - start_time:.3f}s")
    start_time = time.time()

    model = SITCL(cfg)
    logging.info(f"load model done, cost {time.time() - start_time:.3f}s")
    start_time = time.time()

    training_args = TrainingArguments(
        seed=cfg.seed,
        data_seed=cfg.seed,
        full_determinism=True,
        output_dir=save_dir,
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        learning_rate=cfg.other_lr,
        per_device_train_batch_size=cfg.batchsize,
        per_device_eval_batch_size=cfg.batchsize,
        num_train_epochs=cfg.epoch_size,
        logging_dir=os.path.join(log_dir, "tensorboard"),
        logging_strategy="steps",
        log_level="info",
        disable_tqdm=True,
        logging_steps=50,
        report_to="tensorboard",
        fp16=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_safetensors=False,
        max_grad_norm=cfg.max_grad_norm,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_proportion,
        lr_scheduler_type="cosine",
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
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

    train_loader = trainer.get_train_dataloader()

    logging.info(f"train_loader.sampler: {type(train_loader.sampler)}")

    trainer.add_callback(TestPredictionCallback(trainer, dev_set, test_set, pred_dir))

    logging.info(f"trainer start, cost {time.time() - start_time:.3f}s")
    logging.info(
        "[train-start] world_size=%s | process_index=%s | local_rank=%s | train_sampler=%s | sampler_shuffle_each_epoch=%s",
        training_args.world_size,
        training_args.process_index,
        getattr(training_args, "local_rank", None),
        "DistributedSampler(shuffle=True)" if training_args.world_size > 1 else "RandomSampler",
        True,
    )
    trainer.train()

    # best_model_path = os.path.join(save_dir, "best_model")
    # trainer.save_model(best_model_path)
    # logging.info(f"Best model saved to {best_model_path}")

    logging.info("Training finished. No model checkpoint will be saved because save_strategy='no'.")


if __name__ == "__main__":
    main()
