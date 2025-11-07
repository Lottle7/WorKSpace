import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        embedding_size,
        num_layers,
        activation=nn.ReLU(),
        dropout=0.5,
    ):
        super(MLP, self).__init__()
        layers = []
        self.drop = nn.Dropout(dropout)
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, embedding_size))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation)
            layers.append(self.drop)
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation)
                layers.append(self.drop)
            layers.append(nn.Linear(hidden_dim, embedding_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class MLPReWeighting(nn.Module):
    """Use MLP to re-weight all interaction experts."""
    """
        根据输入的多模态 embedding, 通过MLP得到每个专家的权重
        input: 多个模态的embedding (hidden_dim * num_modalities)
        output: 每个专家的权重 (num_branches)
    """

    def __init__(
        self,
        num_modalities,
        num_branches,
        hidden_dim=64,
        hidden_dim_rw=64,
        num_layers=2,
        temperature=0.5,
    ):
        """args:
        hidden_dim: hidden dimension of input embeddings.
        hidden_dim_rw: hidden dimension of the re-weighting model.
        """
        super(MLPReWeighting, self).__init__()
        self.temperature = temperature
        self.mlp = MLP(
            hidden_dim * num_modalities,
            hidden_dim_rw,
            num_branches,
            num_layers,
            activation=nn.ReLU(),
            dropout=0.5,
        )

    def temperature_scaled_softmax(self, logits):
        logits = logits / self.temperature
        return torch.softmax(logits, dim=1)

    def forward(self, inputs):
        if inputs[0].dim() == 3:
            x = [item.mean(dim=1) for item in inputs]
            x = torch.cat(x, dim=1)
        else:
            x = torch.cat(inputs, dim=1)    # (180, num_views*180)
        x = self.mlp(x)
        return self.temperature_scaled_softmax(x)

class InteractionExpert(nn.Module):
    """
    Interaction Expert.
    """

    def __init__(self, fusion_model):
        super(InteractionExpert, self).__init__()
        self.fusion_model = fusion_model

    def forward(self, inputs):
        """
        Forward pass with all modalities present.
        """
        return self._forward_with_replacement(inputs, replace_index=None)

    def forward_with_replacement(self, inputs, replace_index):
        """
        Forward pass with one modality replaced by a random vector.

        Args:
            inputs (list of tensors): List of modality inputs.
            replace_index (int): Index of the modality to replace. If None, no modality is replaced.
        """
        return self._forward_with_replacement(inputs, replace_index=replace_index)

    def _forward_with_replacement(self, inputs, replace_index=None):
        """
        Internal function to handle forward pass with optional modality replacement.
        """
        # Replace specified modality with a random vector
        if replace_index is not None:
            random_vector = torch.rand_like(inputs[replace_index]) # TODO：line 171
            inputs = (
                inputs[:replace_index] + [random_vector] + inputs[replace_index + 1 :]
            )

        region_embs = self.fusion_model(inputs)
        x = region_embs

        return x
    
    def _forward_with_only_one_modality(self, inputs, keep_index):
        """
        只保留一个主模态，其余模态全部用随机向量替换。
        Args:
            inputs (list of tensors): 各模态输入
            keep_index (int): 需要保留的主模态索引，其余模态将被随机替换
        Returns:
            x: 融合后的输出
        """
        # 构造新的输入列表
        new_inputs = []
        for i, inp in enumerate(inputs):
            if i == keep_index:
                new_inputs.append(inp)
            else:
                random_vector = torch.rand_like(inp)   # TODO：line 146
                new_inputs.append(random_vector)
        # 送入融合模型
        out_s, out_t, out_p, out_l = self.fusion_model(new_inputs)
        x = self.fusion_model.out_feature()
        return x

    def forward_multiple(self, inputs):
        """
        Perform (1 + n) forward passes: one with all modalities and one for each modality replaced.

        Args:
            inputs (list of tensors): List of modality inputs.

        Returns:
            List of outputs from the forward passes.
        """
        outputs = []

        outputs.append(self.forward(inputs))    # 先完整输入

        # Forward passes with each modality replaced
        for i in range(len(inputs)):
            outputs.append(self.forward_with_replacement(inputs, replace_index=i)) 
            # outputs.append(self._forward_with_only_one_modality(inputs, keep_index=i))

        return outputs

class RegionMoFE(nn.Module):
    def __init__(
        self,
        num_modalities=3,
        fusion_model=None,
        embedding_size=144,
        hidden_dim=180,
        hidden_dim_rw=64,
        num_layer_rw=2,
        temperature_rw=1,
        triplet_margin=0.8,
    ):
        super(RegionMoFE, self).__init__()
        self.num_branches = num_modalities + 1 + 1  # uni + syn + red
        self.num_modalities = num_modalities
        self.reweight = MLPReWeighting(
            num_modalities,
            self.num_branches,
            hidden_dim=hidden_dim,
            hidden_dim_rw=hidden_dim_rw,
            num_layers=num_layer_rw,
            temperature=temperature_rw,
        )
        self.interaction_experts = nn.ModuleList(
            [
                InteractionExpert(deepcopy(fusion_model))
                for _ in range(self.num_branches)
            ]
        )
        self.triplet_margin = triplet_margin

    def uniqueness_loss_single(self, anchor, pos, neg, margin=1):
        triplet_loss = nn.TripletMarginLoss(margin=margin, p=2, eps=1e-7)
        return triplet_loss(anchor, pos, neg)

    def synergy_loss(self, anchor, negatives):
        total_syn_loss = 0
        anchor_normalized = F.normalize(anchor, p=2, dim=1)
        for negative in negatives:
            negative_normalized = F.normalize(negative, p=2, dim=1)
            cosine_sim = torch.sum(anchor_normalized * negative_normalized, dim=1)
            total_syn_loss += torch.mean(cosine_sim)
        total_syn_loss = total_syn_loss / len(negatives)

        return total_syn_loss  # Synergy loss

    def redundancy_loss(self, anchor, positives):
        total_redundancy_loss = 0
        anchor_normalized = F.normalize(anchor, p=2, dim=1)
        for positive in positives:
            positive_normalized = F.normalize(positive, p=2, dim=1)
            cosine_sim = torch.sum(anchor_normalized * positive_normalized, dim=1)
            total_redundancy_loss += torch.mean(1 - cosine_sim)
        total_redundancy_loss = total_redundancy_loss / len(positives)
        return total_redundancy_loss  # Redundancy loss

    def forward(self, inputs):
        expert_outputs = []

        for expert in self.interaction_experts:
            expert_outputs.append(expert.forward_multiple(inputs))

        ###### Define interaction losses ######
        # First n experts are uniqueness interaction expert
        uniqueness_losses = []
        ### Mask only one modality
        for i in range(self.num_modalities):
            margin = self.triplet_margin

            uniqueness_loss = 0
            outputs = expert_outputs[i]
            anchor = outputs[0]
            neg = outputs[i + 1]    # 主模态被掩码的作为负样本
            positives = outputs[1 : i + 1] + outputs[i + 2 :]    # 其他作为正样本
            for pos in positives:
                uniqueness_loss += self.uniqueness_loss_single(anchor, pos, neg, margin=margin)
            uniqueness_losses.append(uniqueness_loss / len(positives))

        # One Synergy Expert
        synergy_output = expert_outputs[-2]
        synergy_anchor = synergy_output[0]
        synergy_negatives = torch.stack(synergy_output[1:])
        synergy_loss = self.synergy_loss(synergy_anchor, synergy_negatives)

        # One Redundacy Expert
        redundancy_output = expert_outputs[-1]
        redundancy_anchor = redundancy_output[0]
        redundancy_positives = torch.stack(redundancy_output[1:])
        redundancy_loss = self.redundancy_loss(redundancy_anchor, redundancy_positives)

        interaction_losses = uniqueness_losses + [synergy_loss] + [redundancy_loss]

        all_logits = torch.stack([output[0] for output in expert_outputs], dim=1)

        ###### MLP reweighting the experts output ######
        interaction_weights = self.reweight(inputs)  # Get interaction weights
        weights_transposed = interaction_weights.unsqueeze(2)
        weighted_logits = (all_logits * weights_transposed).sum(dim=1)   # (180, 144)


        return (
            expert_outputs,
            interaction_weights,
            weighted_logits,
            interaction_losses,
        )