"""PPED with offline target-conditioned Qwen utterance embeddings.

Prior/posterior evidence selection follows model_sitpcl_12.py. There is no
language model or separate target/type embedding. Both history pools include
an always-valid zero-valued null candidate. Only the prior runs at inference.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeakerAwareEvidenceSelector(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, tau=0.2):
        super().__init__()
        if tau <= 0:
            raise ValueError("selector_tau must be positive")
        self.tau = tau
        self.speaker_emb = nn.Embedding(2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, query, history, valid, speaker_types):
        # Null is appended AFTER speaker enrichment: its key and value are zero.
        keys = history + self.speaker_emb(speaker_types)
        keys = keys.masked_fill(~valid.unsqueeze(-1), 0)
        keys = torch.cat([keys, keys.new_zeros(keys.size(0), 1, keys.size(2))], dim=1)
        mask = torch.cat([valid, valid.new_ones(valid.size(0), 1)], dim=1)
        queries = query.unsqueeze(1).expand_as(keys)
        scores = self.scorer(self.dropout(torch.cat([queries, keys], dim=-1))).squeeze(-1)
        # Float32 probabilities/logs keep KL stable with mixed precision.
        scores = (scores.float() / self.tau).masked_fill(~mask, float('-inf'))
        return scores.softmax(dim=-1)


class CausalHistoryEncoder(nn.Module):
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(hidden_size, hidden_size * 2), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden_size * 2, hidden_size))
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, history, valid):
        history = history.masked_fill(~valid.unsqueeze(-1), 0)
        # Empty rows get one temporary key to avoid all-masked attention NaNs.
        safe_valid = valid.clone()
        safe_valid[:, 0] |= ~valid.any(dim=1)
        width = history.size(1)
        causal_mask = torch.ones(width, width, device=history.device, dtype=torch.bool).triu(1)
        attended, _ = self.self_attn(history, history, history,
                                    attn_mask=causal_mask, key_padding_mask=~safe_valid,
                                    need_weights=False)
        encoded = self.norm1(history + self.dropout(attended))
        encoded = self.norm2(encoded + self.dropout(self.ffn(encoded)))
        return encoded.masked_fill(~valid.unsqueeze(-1), 0)


class SITCL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim, hidden = int(config.embedding_dim), config.hidden_size
        if dim <= 0:
            raise ValueError("embedding_dim must be inferred from the dataset before constructing SITCL")
        if hidden < 2 or hidden % config.num_heads:
            raise ValueError("hidden_size must be >= 2 and divisible by num_heads")
        self.alpha = config.alpha
        self.lambda_distill = config.lambda_distill
        self.posterior_ce_weight = config.posterior_ce_weight
        # Defaults keep the existing Trainer config compatible. Set to 0 on
        # config/model for a no-pair ablation; no new parameters are introduced.
        self.dual_ce_weight = float(getattr(config, 'dual_ce_weight', 1.0))
        if self.dual_ce_weight < 0:
            raise ValueError("dual_ce_weight must be nonnegative")
        self.dual_share_user_history = bool(getattr(config, 'dual_share_user_history', True))
        self.use_history = config.use_history
        self.use_causal_history = config.use_causal_history
        self.utterance_proj = nn.Linear(dim, hidden) if dim != hidden else nn.Identity()
        self.utterance_proj_norm = nn.LayerNorm(hidden)
        self.causal_encoder = CausalHistoryEncoder(hidden, config.num_heads, config.dropout)

        self.dialog_prior_selector = SpeakerAwareEvidenceSelector(hidden, config.dropout, config.selector_tau)
        self.dialog_posterior_selector = SpeakerAwareEvidenceSelector(hidden, config.dropout, config.selector_tau)
        self.user_prior_selector = SpeakerAwareEvidenceSelector(hidden, config.dropout, config.selector_tau)
        self.user_posterior_selector = SpeakerAwareEvidenceSelector(hidden, config.dropout, config.selector_tau)

        self.label_emb = nn.Embedding(config.num_classes, hidden)
        self.posterior_query_proj = nn.Sequential(nn.Linear(hidden * 2, hidden),
                                                   nn.Dropout(config.dropout), nn.GELU())
        # Final utterance and the two history pools; no difference features.
        def classifier():
            return nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(),
                                 nn.Dropout(config.dropout), nn.Linear(hidden, config.num_classes))
        self.main_classifier_prior = classifier()
        self.main_classifier_post = classifier()
        self.cl_dropout = nn.Dropout(config.dropout)
        self.cl_projection_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                                 nn.Linear(hidden, hidden // 2))
        self.register_buffer('global_step', torch.tensor(0))
        self.register_buffer('total_steps', torch.tensor(1))
        self.reset_debug_cache()

    def set_step(self, step, total):
        self.global_step.fill_(step)
        self.total_steps.fill_(max(total, 1))

    def reset_debug_cache(self):
        self._debug_cache = {}
        self._debug_count = 0

    def pop_debug_cache(self):
        result = {key: (value / max(self._debug_count, 1)).item()
                  for key, value in self._debug_cache.items()}
        self.reset_debug_cache()
        return result

    @staticmethod
    def _pool(probabilities, history):
        # The last probability belongs to the zero null value. Omitting that
        # term is exactly equivalent to appending an all-zero value vector.
        return torch.bmm(probabilities[:, :-1].to(history.dtype).unsqueeze(1), history).squeeze(1)

    @staticmethod
    def _features(final, dialog_pool, user_pool):
        return torch.cat([final, dialog_pool, user_pool], dim=-1)

    @staticmethod
    def _kl(prior, posterior):
        teacher = posterior.detach()
        return (teacher * (teacher.clamp_min(1e-12).log() - prior.clamp_min(1e-12).log())).sum(-1).mean()

    def _dual_loss(self, batch, dialog_prob, user_prob):
        """Reuse SOURCE prior probabilities, without detach or a second selector.

        Counterpart embeddings contain its own target. The shared classifier and
        source selector receive gradients from the counterpart's actual label.
        Missing pairs contribute neither loss nor gradients. Called in train only.
        """
        paired = batch['dual_mask']
        vectors = self.utterance_proj(batch['dual_utterance_embeddings'])
        vectors = self.utterance_proj_norm(vectors)
        final = vectors[batch['final_indices']]
        dialog_pool = user_pool = torch.zeros_like(final)
        if self.use_history:
            dialog_valid = batch['dialog_valid'] & paired.unsqueeze(1)
            dialog = vectors[batch['dialog_indices']].masked_fill(~dialog_valid.unsqueeze(-1), 0)
            if self.use_causal_history:
                dialog = self.causal_encoder(dialog, dialog_valid)
            # A target pair does not imply identical cross-conversation user
            # memories. Disable this auxiliary branch unless candidates align.
            user_aligned = paired & batch['dual_user_aligned']
            if not self.dual_share_user_history:
                user_aligned = torch.zeros_like(user_aligned)
            user_valid = batch['user_valid'] & user_aligned.unsqueeze(1)
            users = self.utterance_proj(batch['dual_user_embeddings'])
            users = self.utterance_proj_norm(users)
            if users.size(0):
                user = users[batch['user_indices']].masked_fill(~user_valid.unsqueeze(-1), 0)
            else:
                user = final.new_zeros(*user_valid.shape, final.size(-1))
            dialog_pool = self._pool(dialog_prob, dialog)
            user_pool = self._pool(user_prob, user)
        dual_logits = self.main_classifier_prior(self._features(final, dialog_pool, user_pool))
        # Index before CE: padding labels/vectors never become training targets.
        return F.cross_entropy(dual_logits[paired], batch['dual_labels'][paired])

    def forward(self, **batch):
        utterances = self.utterance_proj(batch['utterance_embeddings'])
        utterances = self.utterance_proj_norm(utterances)
        final = utterances[batch['final_indices']]
        labels = batch.get('label')
        zero = final.new_zeros(())
        distill_loss = posterior_ce_loss = contrastive_loss = zero
        dialog_pool = user_pool = torch.zeros_like(final)
        dialog_prob = user_prob = final.new_ones(final.size(0), 1)

        if self.use_history:
            dialog_valid, user_valid = batch['dialog_valid'], batch['user_valid']
            dialog_history = utterances[batch['dialog_indices']]
            dialog_history = dialog_history.masked_fill(~dialog_valid.unsqueeze(-1), 0)
            if self.use_causal_history:
                # Do not encode the final utterance with history: otherwise
                # selecting null could not remove the history's influence.
                dialog_history = self.causal_encoder(dialog_history, dialog_valid)
            user_vectors = self.utterance_proj(batch['user_embeddings'])
            user_vectors = self.utterance_proj_norm(user_vectors)
            if user_vectors.size(0):
                user_history = user_vectors[batch['user_indices']]
                user_history = user_history.masked_fill(~user_valid.unsqueeze(-1), 0)
            else:
                user_history = final.new_zeros(*user_valid.shape, final.size(-1))
            dialog_prob = self.dialog_prior_selector(final, dialog_history, dialog_valid, batch['dialog_speaker_types'])
            user_prob = self.user_prior_selector(final, user_history, user_valid, batch['user_speaker_types'])
            dialog_pool = self._pool(dialog_prob, dialog_history)
            user_pool = self._pool(user_prob, user_history)

            # Gold information belongs exclusively to the training posterior.
            if self.training and labels is not None and batch.get('all_labels') is not None:
                query = self.posterior_query_proj(torch.cat([final, self.label_emb(labels)], dim=-1))
                dialog_post = (dialog_history + self.label_emb(batch['dialog_labels'])).masked_fill(
                    ~dialog_valid.unsqueeze(-1), 0)
                # Missing user labels are not silently interpreted as class 0.
                known = torch.tensor([x is not None for x in batch['last_speaker_sentences_labels']],
                                     device=final.device, dtype=torch.bool).unsqueeze(1)
                user_label_repr = self.label_emb(batch['user_labels']) * known.unsqueeze(-1)
                user_post = (user_history + user_label_repr).masked_fill(~user_valid.unsqueeze(-1), 0)
                dialog_teacher = self.dialog_posterior_selector(query, dialog_post, dialog_valid, batch['dialog_speaker_types'])
                user_teacher = self.user_posterior_selector(query, user_post, user_valid, batch['user_speaker_types'])
                distill_loss = self._kl(dialog_prob, dialog_teacher) + self._kl(user_prob, user_teacher)
                # Preserve original PPED: posterior pools label-augmented values;
                # the null value remains zero and never receives a label vector.
                post_features = self._features(final, self._pool(dialog_teacher, dialog_post),
                                               self._pool(user_teacher, user_post))
                posterior_ce_loss = F.cross_entropy(self.main_classifier_post(post_features), labels)

        logits = self.main_classifier_prior(self._features(final, dialog_pool, user_pool))
        if labels is None:
            return {'logits': logits}
        ce_loss = F.cross_entropy(logits, labels)
        if self.training and self.alpha > 0 and batch.get('all_labels') is not None:
            if self.use_history:
                representations = [torch.cat([dialog_history[i, dialog_valid[i]], final[i:i+1]], dim=0)
                                   for i in range(final.size(0))]
            else:
                representations = [utterances[st:ed] for st, ed in batch['dia_idx']]
            contrastive_loss = self._compute_supcon_loss(representations, batch['target'], batch['all_labels'])
        dual_ce_loss = zero
        if (self.training and self.dual_ce_weight > 0
                and batch.get('dual_mask') is not None and batch['dual_mask'].any()):
            dual_ce_loss = self._dual_loss(batch, dialog_prob, user_prob)
        alpha = self._get_alpha() if self.alpha > 0 else 0.0
        loss = ce_loss + self.lambda_distill * distill_loss + self.posterior_ce_weight * posterior_ce_loss + alpha * contrastive_loss + self.dual_ce_weight * dual_ce_loss
        if self.training:
            pair_fraction = batch['dual_mask'].float().mean() if 'dual_mask' in batch else zero
            user_fraction = ((batch['dual_mask'] & batch['dual_user_aligned']).float().mean()
                             if self.dual_share_user_history and 'dual_mask' in batch else zero)
            values = dict(loss=loss, ce_loss=ce_loss, distill_loss=distill_loss,
                          dual_ce_loss=dual_ce_loss, dual_pair_fraction=pair_fraction,
                          dual_user_aligned_fraction=user_fraction,
                          posterior_ce_loss=posterior_ce_loss, target_contrastive_loss=contrastive_loss,
                          dialog_null_prob=dialog_prob[:, -1].mean(), user_null_prob=user_prob[:, -1].mean())
            for key, value in values.items():
                self._debug_cache[key] = self._debug_cache.get(key, 0) + value.detach()
            self._debug_count += 1
        return {'loss': loss, 'logits': logits}

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
