#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: xiongjie <xiongjiezk@163.com>
# @date: 2026/5/16

"""SITCLv5

Posterior-guided latent evidence stance model with dialogue sequential dependency.

Core design:
- Split evidence into two pools: dialog history pool and user history pool.
- Posterior selectors use label-augmented query/evidence to guide prior selectors.
- Prior selectors are used at inference time.
- Final classifier only consumes evidence representations and target semantics,
  without injecting label information.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


def map_sequence(input_list):
    if torch.is_tensor(input_list):
        if input_list.dim() == 0:
            input_list = [input_list.item()]
        else:
            input_list = input_list.tolist()
    elif not isinstance(input_list, (list, tuple)):
        input_list = [input_list]

    num_to_order = {}
    order = 0
    result = []
    for num in input_list:
        if num not in num_to_order:
            num_to_order[num] = order
            order += 1
        result.append(num_to_order[num])
    return result


class SpeakerAwareEvidenceSelector(nn.Module):
    """Lightweight speaker-aware selector for one evidence pool."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1, tau: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.speaker_emb = nn.Embedding(2, hidden_dim)  # 0: same, 1: different
        self.dropout = nn.Dropout(dropout)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, query_repr, history_repr, query_speaker, hist_speakers, target_repr):
        # query_repr: (D)
        # history_repr: (L, D)
        L = history_repr.size(0)
        if L == 0:
            return torch.empty(0, device=query_repr.device)

        if not torch.is_tensor(hist_speakers):
            hist_speakers = torch.as_tensor(hist_speakers, device=query_repr.device)
        hist_speakers = hist_speakers.reshape(-1)
        if hist_speakers.numel() == 0:
            return torch.empty(0, device=query_repr.device)
        
        # 0: same speaker, 1: different speaker
        speaker_type = (hist_speakers != query_speaker).long()
        speaker_feat = self.speaker_emb(speaker_type)

        q_expanded = query_repr.unsqueeze(0).expand(L, -1)
        target_expanded = target_repr.unsqueeze(0).expand(L, -1)
        q_expanded_conditioned = q_expanded + target_expanded
        history_repr_conditioned = history_repr + speaker_feat
        features = torch.cat([q_expanded_conditioned, history_repr_conditioned], dim=-1)
        logits = self.scorer(self.dropout(features)).squeeze(-1)
        probs = torch.softmax(logits / self.tau, dim=-1)
        return probs


class CausalHistoryEncoder(nn.Module):
    """Select useful historical utterances with causal attention."""

    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq_repr):
        seq = seq_repr.unsqueeze(0)
        L = seq.size(1)
        causal_mask = torch.triu(torch.ones(L, L, device=seq_repr.device, dtype=torch.bool), diagonal=1)
        attn_out, attn_weights = self.self_attn(seq, seq, seq, attn_mask=causal_mask, need_weights=True, average_attn_weights=False)
        seq_out = self.norm1(seq + self.dropout(attn_out)).squeeze(0)
        seq_out = self.norm2(seq_out + self.dropout(self.ffn(seq_out)))
        return seq_out, attn_weights.squeeze(0)


class SITCL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.selector_tau = getattr(config, "selector_tau", 0.2)
        self.dropout = getattr(config, "dropout", 0.1)
        self.alpha = getattr(config, "alpha", 1.0)
        self.lambda_distill = getattr(config, "lambda_distill", 1.0)
        self.posterior_ce_weight = getattr(config, "posterior_ce_weight", 1.0)

        self.bert = AutoModel.from_pretrained(config.bert_dir)
        self.utterance_proj = nn.Linear(self.bert.config.hidden_size, self.hidden_size) if self.bert.config.hidden_size != self.hidden_size else nn.Identity()

        self.causal_encoder = CausalHistoryEncoder(self.hidden_size, num_heads=getattr(config, "num_heads", 4), dropout=getattr(config, "dropout", 0.1))

        self.dialog_prior_selector = SpeakerAwareEvidenceSelector(self.hidden_size, dropout=self.dropout, tau=self.selector_tau)
        self.dialog_posterior_selector = SpeakerAwareEvidenceSelector(self.hidden_size, dropout=self.dropout, tau=self.selector_tau)
        self.user_prior_selector = SpeakerAwareEvidenceSelector(self.hidden_size, dropout=self.dropout, tau=self.selector_tau)
        self.user_posterior_selector = SpeakerAwareEvidenceSelector(self.hidden_size, dropout=self.dropout, tau=self.selector_tau)

        self.posterior_query_proj = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Dropout(self.dropout), 
            nn.GELU(),
        )

        self.main_classifier_prior = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, config.num_classes),
        )
        self.main_classifier_post = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, config.num_classes),
        )
        self.target_type_emb = nn.Embedding(4, self.hidden_size)
        self.label_emb = nn.Embedding(config.num_classes, self.hidden_size)

        self.cl_dropout = nn.Dropout(self.dropout)
        self.cl_projection_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
        )

        self.criterion = nn.CrossEntropyLoss()
        self._debug_cache: Dict[str, List[float]] = {}
        self.reset_debug_cache()

        self.register_buffer("global_step", torch.zeros(1))
        self.register_buffer("total_steps", torch.ones(1))

    def set_step(self, step, total):
        self.global_step.fill_(step)
        self.total_steps.fill_(total)

    def _weighted_compute_supcon_loss(self, H_lst, targets, all_labels, utterance_weights=None):
        """
        计算实例级的监督对比学习损失 (Supervised Contrastive Learning)
        H_lst: List[Tensor], 每个 Tensor shape 为 (L_i, D)，代表一个对话中的所有话语
        targets: List[str], 长度为 Batch Size
        all_labels: List[List[int]], 包含对话中所有话语的真实立场标签
        utterance_weights: List[Tensor | List[float]]，与 H_lst 对应的 utterance 权重
        """
        if not self.training or all_labels is None:
            return torch.tensor(0.0, device=H_lst[0].device)

        features = []
        flat_targets = []
        flat_stances = []
        flat_weights = []

        # 展平 Batch 内所有的话语级别特征
        for i, h in enumerate(H_lst):
            L_i = h.size(0)
            features.append(h)
            # 每个话语继承所属对话的 target
            flat_targets.extend([targets[i]] * L_i)
            # 获取对应的细粒度立场标签
            flat_stances.extend(all_labels[i])
            if utterance_weights is None:
                flat_weights.extend([1.0] * L_i)
            else:
                w = utterance_weights[i]
                if torch.is_tensor(w):
                    flat_weights.extend(w.detach().float().tolist())
                else:
                    flat_weights.extend([float(x) for x in w])

        features = torch.cat(features, dim=0)  # (N, D)
        device = features.device
        N = features.size(0)

        # 1. 特征增强与独立空间映射 (Projection)
        features_aug = self.cl_dropout(features)
        z = self.cl_projection_head(features_aug)  # (N, D_proj)
        z = F.normalize(z, p=2, dim=-1)

        # 2. 计算相似度矩阵
        tau = getattr(self.config, "tau", 0.07) # 对比学习通常用更低的温度系数，增加难度
        sim_matrix = torch.matmul(z, z.T) / tau  # (N, N)

        # 3. 构造 SupCon 掩码
        target_dict = {t: idx for idx, t in enumerate(set(flat_targets))}
        target_ids = torch.tensor([target_dict[t] for t in flat_targets], device=device)
        stance_ids = torch.tensor(flat_stances, device=device)
        weight_tensor = torch.tensor(flat_weights, device=device, dtype=features.dtype).clamp_min(0.0)

        # Target 相同 且 Stance 相同 -> 严格的正样本对 (Mask=1)
        # Target 相同 但 Stance 相反 -> Mask=0，自然成为分母中的 Hard Negative
        target_mask = (target_ids.unsqueeze(0) == target_ids.unsqueeze(1))
        stance_mask = (stance_ids.unsqueeze(0) == stance_ids.unsqueeze(1))
        pos_mask = target_mask & stance_mask

        # 排除自身 (Self-contrast)
        self_mask = torch.eye(N, dtype=torch.bool, device=device)
        pos_mask = pos_mask.masked_fill(self_mask, False)

        # 4. 数值稳定的 InfoNCE 计算
        # 减去最大值防止 exp 溢出
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix_shifted = sim_matrix - sim_max.detach()

        exp_sim = torch.exp(sim_matrix_shifted)
        exp_sim = exp_sim.masked_fill(self_mask, 0.0)  # 分母中也不包含自身

        # 计算 log(分子 / 分母) = 分子 - log(分母)
        log_prob = sim_matrix_shifted - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        # 5. 过滤掉没有正样本的锚点，聚合 Loss
        valid_anchors = pos_mask.sum(dim=1) > 0
        if not valid_anchors.any():
            return torch.tensor(0.0, device=device)

        pos_mask_valid = pos_mask[valid_anchors]
        log_prob_valid = log_prob[valid_anchors]
        anchor_weights = weight_tensor[valid_anchors]

        # 仅对正样本求均值: sum(mask * log_prob) / sum(mask)
        mean_log_prob_pos = (pos_mask_valid * log_prob_valid).sum(dim=1) / pos_mask_valid.sum(dim=1)
        weighted_loss = -(mean_log_prob_pos * anchor_weights)
        loss = weighted_loss.mean()
        return loss

    def _compute_supcon_loss(self, H_lst, targets, all_labels):
        """
        计算实例级的监督对比学习损失 (Supervised Contrastive Learning)
        规则：stance 相同为正样本；同 target 的正样本权重更高，不同 target 的正样本权重更低。

        H_lst: List[Tensor], 每个 Tensor shape 为 (L_i, D)，代表一个对话中的所有话语
        targets: List[str], 长度为 Batch Size
        all_labels: List[List[int]], 包含对话中所有话语的真实立场标签
        """
        if not self.training or all_labels is None:
            return torch.tensor(0.0, device=H_lst[0].device)

        features = []
        flat_targets = []
        flat_stances = []

        # 展平 Batch 内所有的话语级别特征
        for i, h in enumerate(H_lst):
            L_i = h.size(0)
            features.append(h)
            flat_targets.extend([targets[i]] * L_i)
            flat_stances.extend(all_labels[i])

        features = torch.cat(features, dim=0)  # (N, D)
        device = features.device
        N = features.size(0)

        # 1. 特征增强与独立空间映射 (Projection)
        features_aug = self.cl_dropout(features)
        z = self.cl_projection_head(features_aug)  # (N, D_proj)
        z = F.normalize(z, p=2, dim=-1)

        # 2. 计算相似度矩阵
        tau = getattr(self.config, "tau", 0.07) # 对比学习通常用更低的温度系数，增加难度
        sim_matrix = torch.matmul(z, z.T) / tau  # (N, N)

        target_dict = {t: idx for idx, t in enumerate(set(flat_targets))}
        target_ids = torch.tensor([target_dict[t] for t in flat_targets], device=device)
        stance_ids = torch.tensor(flat_stances, device=device)

        same_stance_mask = stance_ids.unsqueeze(0) == stance_ids.unsqueeze(1)
        same_target_mask = target_ids.unsqueeze(0) == target_ids.unsqueeze(1)
        self_mask = torch.eye(N, dtype=torch.bool, device=device)

        pos_weight = torch.where(same_target_mask, torch.ones_like(sim_matrix), torch.full_like(sim_matrix, 0.5))
        pos_weight = pos_weight * same_stance_mask.float()
        pos_weight = pos_weight.masked_fill(self_mask, 0.0)

        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix_shifted = sim_matrix - sim_max.detach()

        exp_sim = torch.exp(sim_matrix_shifted)
        exp_sim = exp_sim.masked_fill(self_mask, 0.0)
        log_prob = sim_matrix_shifted - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        valid_anchors = pos_weight.sum(dim=1) > 0
        if not valid_anchors.any():
            return torch.tensor(0.0, device=device)

        pos_weight_valid = pos_weight[valid_anchors]
        log_prob_valid = log_prob[valid_anchors]
        weighted_pos_sum = (pos_weight_valid * log_prob_valid).sum(dim=1)
        weight_norm = pos_weight_valid.sum(dim=1).clamp_min(1.0)
        mean_log_prob_pos = weighted_pos_sum / weight_norm
        loss = -mean_log_prob_pos.mean()
        return loss


    def _compute_supcon_loss_stance_only(self, H_lst, all_labels):
        """
        计算实例级的监督对比学习损失 (Supervised Contrastive Learning)
        规则：只要 stance 相同就视为正样本，不再约束 target 必须相同。

        H_lst: List[Tensor], 每个 Tensor shape 为 (L_i, D)，代表一个对话中的所有话语
        targets: List[str], 长度为 Batch Size（保留参数以兼容旧接口）
        all_labels: List[List[int]], 包含对话中所有话语的真实立场标签
        """
        if not self.training or all_labels is None:
            return torch.tensor(0.0, device=H_lst[0].device)

        features = []
        flat_stances = []

        # 展平 Batch 内所有的话语级别特征
        for i, h in enumerate(H_lst):
            L_i = h.size(0)
            features.append(h)
            flat_stances.extend(all_labels[i])

        features = torch.cat(features, dim=0)  # (N, D)
        device = features.device
        N = features.size(0)

        # 1. 特征增强与独立空间映射 (Projection)
        features_aug = self.cl_dropout(features)
        z = self.cl_projection_head(features_aug)  # (N, D_proj)
        z = F.normalize(z, p=2, dim=-1)

        # 2. 计算相似度矩阵
        tau = getattr(self.config, "tau", 0.07) # 对比学习通常用更低的温度系数，增加难度
        sim_matrix = torch.matmul(z, z.T) / tau  # (N, N)

        # 3. 构造 SupCon 掩码：只要 stance 相同就是正样本
        stance_ids = torch.tensor(flat_stances, device=device)
        pos_mask = stance_ids.unsqueeze(0) == stance_ids.unsqueeze(1)

        # 排除自身 (Self-contrast)
        self_mask = torch.eye(N, dtype=torch.bool, device=device)
        pos_mask = pos_mask.masked_fill(self_mask, False)

        # 4. 数值稳定的 InfoNCE 计算
        # 减去最大值防止 exp 溢出
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix_shifted = sim_matrix - sim_max.detach()

        exp_sim = torch.exp(sim_matrix_shifted)
        exp_sim = exp_sim.masked_fill(self_mask, 0.0)  # 分母中也不包含自身

        # 计算 log(分子 / 分母) = 分子 - log(分母)
        log_prob = sim_matrix_shifted - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        # 5. 过滤掉没有正样本的锚点，聚合 Loss
        valid_anchors = pos_mask.sum(dim=1) > 0
        if not valid_anchors.any():
            return torch.tensor(0.0, device=device)

        pos_mask_valid = pos_mask[valid_anchors]
        log_prob_valid = log_prob[valid_anchors]

        # 仅对正样本求均值: sum(mask * log_prob) / sum(mask)
        mean_log_prob_pos = (pos_mask_valid * log_prob_valid).sum(dim=1) / pos_mask_valid.sum(dim=1).clamp_min(1.0)
        loss = -mean_log_prob_pos.mean()
        return loss

    def encode_utterances(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_token = out.pooler_output
        return self.utterance_proj(cls_token)

    @staticmethod
    def _entropy_from_logits(logits):
        p = torch.softmax(logits, dim=-1)
        return (-p * torch.log(p.clamp_min(1e-12))).sum(dim=-1)

    def reset_debug_cache(self):
        self._debug_cache = {
            "loss": [],
            "ce_loss": [],
            "distill_loss": [],
            "target_contrastive_loss": [],
            "posterior_ce_loss": [],
            "main_confidence": [],
            "main_entropy": [],
            "posterior_count": [],
            "history_len": [],
            "dialog_prior_prob_max": [],
            "user_prior_prob_max": [],
            "dialog_posterior_prob_max": [],

            "prior_pool_norm": [],
            "posterior_pool_norm": [],
            "target_norm": [],
        }

    def _cache_scalar(self, key, value):
        self._debug_cache.setdefault(key, []).append(float(value))

    def pop_debug_cache(self):
        out = {k: (sum(v) / len(v) if len(v) > 0 else 0.0) for k, v in self._debug_cache.items()}
        self.reset_debug_cache()
        return out

    @staticmethod
    def _target_type_to_id(target_type):
        type_ids = []
        for tp in target_type:
            if tp in ("c", "claim"):
                type_ids.append(1)
            elif tp in ("n", "noun", "noun_phrase"):
                type_ids.append(2)
            else:
                type_ids.append(0)
        return type_ids

    def _gen_user_sentences_idx(self, counts):
        user_idx = []
        start = 0
        for count in counts:
            user_idx.append((start, start + count))
            start += count
        return user_idx

    @staticmethod
    def _safe_pool(prob, evidence):
        if evidence.size(0) == 0 or prob.numel() == 0:
            return torch.zeros(evidence.size(-1), device=evidence.device)
        return torch.matmul(prob.unsqueeze(0), evidence).squeeze(0)

    def _get_alpha(self):
        step = float(self.global_step.item())
        total = max(float(self.total_steps.item()), 1.0)
        r = step / total

        alpha_min = 0.3
        alpha_peak = self.alpha
        if alpha_peak < alpha_min:
            return alpha_peak
        warmup_ratio = getattr(self.config, "alpha_warmup_ratio", 0.1)
        decay_ratio = getattr(self.config, "alpha_decay_ratio", 0.8)

        if r < warmup_ratio:
            # 0.3 -> alpha_peak
            t = r / max(warmup_ratio, 1e-12)
            return alpha_min + (alpha_peak - alpha_min) * t

        if r < decay_ratio:
            # alpha_peak -> 0.3
            t = (r - warmup_ratio) / max(decay_ratio - warmup_ratio, 1e-12)
            return alpha_peak - (alpha_peak - alpha_min) * t

        return alpha_min

    def forward(self, **kwargs):

        step = int(self.global_step.item())
        total = int(self.total_steps.item())
        ratio = step / max(total, 1)

        input_ids = kwargs["input_ids"]
        attention_mask = kwargs["attention_mask"]
        token_type_ids = kwargs["token_type_ids"]
        label = kwargs["label"]
        dia_idx = kwargs["dia_idx"]
        all_labels = kwargs.get("all_labels", None)
        speakers = kwargs["speakers"]
        target_input_ids = kwargs["target_input_ids"]
        target_attention_mask = kwargs["target_attention_mask"]
        target_token_type_ids = kwargs["target_token_type_ids"]
        target_type = kwargs.get("target_type", [""] * len(dia_idx))
        targets = kwargs["target"]

        utter_repr = self.encode_utterances(input_ids, attention_mask, token_type_ids)
        target_repr_batch = self.encode_utterances(target_input_ids, target_attention_mask, target_token_type_ids)
        type_ids = self._target_type_to_id(target_type)
        type_emb = self.target_type_emb(torch.tensor(type_ids, device=target_repr_batch.device))
        target_cond = target_repr_batch + type_emb

        final_reprs = []
        H_final = []
        posterior_kl_dialog_sum = torch.tensor(0.0, device=utter_repr.device)
        posterior_ce_loss_sum = torch.tensor(0.0, device=utter_repr.device)
        posterior_count = 0
        prior_pool_norms = []
        posterior_pool_norms = []
        dialog_prior_prob_max_list = []
        dialog_posterior_prob_max_list = []
        history_lens = []

        for i, (dia_st, dia_ed) in enumerate(dia_idx):
            u = utter_repr[dia_st:dia_ed]
            causal_u, _ = self.causal_encoder(u)
            H_final.append(causal_u)
            final_utt = causal_u[-1]
            history_repr = causal_u[:-1]
            history_len = history_repr.size(0)
            current_target = target_cond[i]
            speakers_i = torch.tensor(speakers[i], device=u.device, dtype=torch.long)

            query_speaker = speakers_i[-1]
            dialog_history = history_repr
            dialog_speakers = speakers_i[:-1]

            dialog_prior_prob = self.dialog_prior_selector(final_utt, dialog_history, query_speaker, dialog_speakers, current_target)
            dialog_pool = self._safe_pool(dialog_prior_prob, dialog_history) if dialog_history.size(0) > 0 else torch.zeros_like(final_utt)

            prior_pool_norms.append(dialog_pool.norm(p=2).detach().item())
            dialog_prior_prob_max = dialog_prior_prob.max().detach().item() if dialog_prior_prob.numel() > 0 else 0.0
            dialog_prior_prob_max_list.append(dialog_prior_prob_max)
            history_lens.append(float(dialog_history.size(0)))

            interaction_dialog = torch.abs(final_utt - dialog_pool)
            final_repr_prior = torch.cat([
                final_utt,
                dialog_pool,
                interaction_dialog,
                current_target,
            ], dim=-1)

            final_reprs.append(final_repr_prior)

            if all_labels is not None and history_len > 0:
                final_label = int(label[i].item())
                hist_labels = torch.tensor(all_labels[i][:-1], device=u.device, dtype=torch.long)
                hist_label_proto = self.label_emb(hist_labels)
                query_label_proto = self.label_emb(torch.tensor(final_label, device=u.device, dtype=torch.long))
                posterior_query = self.posterior_query_proj(torch.cat([final_utt, query_label_proto], dim=-1))
                dialog_history_post = dialog_history + hist_label_proto if dialog_history.size(0) > 0 else dialog_history
                dialog_posterior_prob = self.dialog_posterior_selector(posterior_query, dialog_history_post, query_speaker, dialog_speakers, current_target)
                if dialog_prior_prob.numel() > 0 and dialog_posterior_prob.numel() > 0:
                    kl_dialog = torch.sum(dialog_posterior_prob.detach() * (
                        torch.log(dialog_posterior_prob.detach().clamp_min(1e-12)) - torch.log(dialog_prior_prob.clamp_min(1e-12))
                    ))
                    posterior_kl_dialog_sum = posterior_kl_dialog_sum + kl_dialog
                posterior_count += 1
                dialog_posterior_prob_max = dialog_posterior_prob.max().detach().item() if dialog_posterior_prob.numel() > 0 else 0.0
                dialog_posterior_prob_max_list.append(dialog_posterior_prob_max)

                dialog_pool_post = self._safe_pool(dialog_posterior_prob, dialog_history_post) if dialog_history_post.size(0) > 0 else torch.zeros_like(final_utt)
                posterior_pool_norms.append(dialog_pool_post.norm(p=2).detach().item())

                interaction_dialog_post = torch.abs(final_utt - dialog_pool_post)
                final_repr_post = torch.cat([
                    final_utt,
                    dialog_pool_post,
                    interaction_dialog_post,
                    current_target,
                ], dim=-1)

                # 不混入主批次，单独过分类器算辅助 Loss
                logits_post = self.main_classifier_post(final_repr_post.unsqueeze(0)).squeeze(0)
                ce_loss_post = self.criterion(logits_post.unsqueeze(0), label[i].unsqueeze(0))
                posterior_ce_loss_sum += ce_loss_post
            
        final_reprs = torch.stack(final_reprs, dim=0)
        logits = self.main_classifier_prior(final_reprs)
        ce_loss = self.criterion(logits, label)

        posterior_kl_dialog = posterior_kl_dialog_sum / max(posterior_count, 1)
        posterior_ce_loss = posterior_ce_loss_sum / max(posterior_count, 1)
        target_contrastive_loss = self._compute_supcon_loss(H_final, targets, all_labels)
        distill_loss = posterior_kl_dialog
        alpha_t = self._get_alpha()

        loss = (
            ce_loss
            + self.lambda_distill * distill_loss
            + alpha_t * target_contrastive_loss
            + self.posterior_ce_weight * posterior_ce_loss
        )

        with torch.no_grad():
            main_conf = torch.softmax(logits, dim=-1).max(dim=-1).values.mean().item()
            main_ent = self._entropy_from_logits(logits).mean().item()
            self._cache_scalar("loss", loss.item())
            self._cache_scalar("ce_loss", ce_loss.item())
            self._cache_scalar("distill_loss", distill_loss.item())
            self._cache_scalar("target_contrastive_loss", target_contrastive_loss.item())
            self._cache_scalar("posterior_ce_loss", posterior_ce_loss.item())
            self._cache_scalar("main_confidence", main_conf)
            self._cache_scalar("main_entropy", main_ent)
            self._cache_scalar("posterior_count", float(posterior_count))
            self._cache_scalar("history_len", sum(history_lens) / len(history_lens) if len(history_lens) > 0 else 0.0)
            self._cache_scalar("dialog_prior_prob_max", sum(dialog_prior_prob_max_list) / len(dialog_prior_prob_max_list) if len(dialog_prior_prob_max_list) > 0 else 0.0)
            self._cache_scalar("dialog_posterior_prob_max", sum(dialog_posterior_prob_max_list) / len(dialog_posterior_prob_max_list) if len(dialog_posterior_prob_max_list) > 0 else 0.0)
            self._cache_scalar("prior_pool_norm", sum(prior_pool_norms) / len(prior_pool_norms) if len(prior_pool_norms) > 0 else 0.0)
            self._cache_scalar("posterior_pool_norm", sum(posterior_pool_norms) / len(posterior_pool_norms) if len(posterior_pool_norms) > 0 else 0.0)
            self._cache_scalar("target_norm", target_cond.norm(p=2, dim=-1).mean().item())

        return {"loss": loss, "logits": logits, "labels": label}
