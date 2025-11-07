import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import GeographicAdjacencyCalculator
from parse_args import args
from Encoder import Encoder

class DeepFc(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DeepFc, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            nn.Linear(input_dim * 2, output_dim),
            nn.LeakyReLU(negative_slope=0.3, inplace=True), )

        self.output = None

    def forward(self, x):
        output = self.model(x)
        self.output = output
        return output

    def out_feature(self):
        return self.output


class ViewFusion(nn.Module):
    def __init__(self, emb_dim, out_dim):
        super(ViewFusion, self).__init__()
        self.W = nn.Conv1d(emb_dim, out_dim, kernel_size=1, bias=False)
        self.f1 = nn.Conv1d(out_dim, 1, kernel_size=1)
        self.f2 = nn.Conv1d(out_dim, 1, kernel_size=1)
        self.act = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, src):
        seq_fts = self.W(src)
        f_1 = self.f1(seq_fts)
        f_2 = self.f2(seq_fts)
        logits = f_1 + f_2.transpose(1, 2)
        coefs = torch.mean(self.act(logits), dim=-1)
        coefs = torch.mean(coefs, dim=0)
        coefs = F.softmax(coefs, dim=-1)
        return coefs

class GraphormerBlock(nn.Module):
    def __init__(self, input_dim, nhead, dropout, dim_feedforward=2048, spatial_alpha=0.1):
        super(GraphormerBlock, self).__init__()
        
        # 空间编码层
        self.spatial_encoding = nn.Linear(1, 1)
        self.centrality_proj = nn.Linear(1, input_dim)  # 中心性投影
        self.spatial_alpha = nn.Parameter(torch.tensor(spatial_alpha))  # 中心性注意力偏置强度
        
        # 多头注意力
        self.self_attn = nn.MultiheadAttention(input_dim, nhead, dropout=dropout, batch_first=True)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, input_dim)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, adj_matrix=None):
        if adj_matrix is not None:
            ### 空间编码（可选）
            adj_encoded = self.spatial_encoding(adj_matrix.unsqueeze(-1))  # [N,N,1]
            spatial_bias = adj_encoded.squeeze(-1)

            ### 节点中心性特征（加权度/强度）
            deg = adj_matrix.sum(-1, keepdim=True)                         # [N,1]
            deg = deg / (deg.max() + 1e-6)
            cent_feat = self.centrality_proj(deg)                           # [N,D]
            if src.dim() == 3:
                cent_feat = cent_feat.unsqueeze(0)                          # [1,N,D]
            src = src + cent_feat                                           # 节点级注入

            ### 注意力偏置
            attn_bias = self.spatial_alpha * spatial_bias                      # [N,N]
        else:
            attn_bias = None
        
        # 自注意力
        src = self.norm1(src)
        attn_output, _ = self.self_attn(src, src, src, attn_mask=attn_bias)
        src = src + self.dropout(attn_output)
        
        # 前馈网络
        src = self.norm2(src)
        ffn_output = self.ffn(src)
        src = src + self.dropout(ffn_output)
        
        return src

class GraphormerRegionFusion(nn.Module):
    def __init__(self, input_dim, num_graphormer, no_head, dropout, spatial_alpha):
        super(GraphormerRegionFusion, self).__init__()
        self.input_dim = input_dim
        self.num_block = num_graphormer
        NO_head = no_head
        dropout = dropout

        # 多个GraphormerBlock
        self.blocks = nn.ModuleList([
            GraphormerBlock(input_dim=input_dim, nhead=NO_head, dropout=dropout, spatial_alpha=spatial_alpha) 
            for _ in range(self.num_block)
        ])

        # 输出层
        self.fc = DeepFc(input_dim, input_dim)

    def forward(self, x, adj_matrix=None):
        """
        Args:
            x: [batch_size, num_regions, input_dim] 区域特征
            adj_matrix: [num_regions, num_regions] 邻接矩阵
        """
        out = x
        
        # 依次通过每个GraphormerBlock
        for block in self.blocks:
            out = block(out, adj_matrix)
        
        # 移除batch维度（如果存在）
        if out.dim() == 3 and out.size(0) == 1:
            out = out.squeeze(0)  # [1, num_regions, input_dim] -> [num_regions, input_dim]
        
        # 通过输出层
        out = self.fc(out)
        
        return out  # [num_regions, input_dim]

class FusionExpert(nn.Module):
    def __init__(self, POI_dim, landUse_dim, c, input_dim, output_dim, d_prime, num_graphormer, no_head, dropout, spatial_alpha):
        super(FusionExpert, self).__init__()
        self.input_dim = input_dim
        self.encoder = Encoder(POI_dim, landUse_dim, input_dim, c)

        self.regionFusionLayer = GraphormerRegionFusion(input_dim, num_graphormer, no_head, dropout, spatial_alpha)

        self.fc = DeepFc(input_dim, output_dim)

        self.viewFusionLayer = ViewFusion(input_dim, d_prime)

        self.geo_calculator = GeographicAdjacencyCalculator(
            distance_type='haversine',  
            normalization='gaussian',   
            threshold=10.0
        )

        if args.region_file_path:
            file_path = args.data_path + args.region_file_path
            print(f"Loading region file from: {file_path}")
            self.adjacency_matrix = self.geo_calculator.compute_adjacency_matrix(file_path)
            self.adjacency_matrix = torch.tensor(self.adjacency_matrix, dtype=torch.float32)
        else:
            self.adjacency_matrix = None

        self.decoder_s = nn.Linear(output_dim, output_dim)  
        self.decoder_t = nn.Linear(output_dim, output_dim)
        self.decoder_p = nn.Linear(output_dim, output_dim)  
        self.decoder_l = nn.Linear(output_dim, output_dim)

    def get_adjacency_matrix(self):
        """get adjacency matrix"""
        if self.adjacency_matrix is not None:
            device = next(self.parameters()).device
            return self.adjacency_matrix.to(device)
        else:
            device = next(self.parameters()).device
            num_regions = 180 
            return torch.eye(num_regions, device=device)

    def forward(self, x):
        x = torch.stack(x)
        adj_matrix = self.get_adjacency_matrix()

        # ---------------------------------------------

        out1 = x.transpose(0, 2)  # (region_num, region_num, num_views)
        coef = self.viewFusionLayer(out1)
        temp_out = torch.zeros(args.region_num, args.region_num).to(args.device)
        for i in range(len(x)):
            temp_out += coef[i] * x[i]

        # --------------------------------------------------

        temp_out = temp_out[np.newaxis]  # (1, region_num, region_num)
        temp_out = self.regionFusionLayer(temp_out, adj_matrix)  # (region_num, region_num)
        x = self.fc(temp_out)  # (region_num, 144)

        return x