import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from parse_args import args
from torchvision import models

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
    
class intraAFL_Block(nn.Module):

    def __init__(self, input_dim, nhead, c, dropout, dim_feedforward=2048):
        super(intraAFL_Block, self).__init__()
        self.self_attn = nn.MultiheadAttention(input_dim, nhead, dropout=dropout, batch_first=True, bias=True)
        self.dropout = nn.Dropout(dropout)

        self.linear1 = nn.Linear(input_dim, dim_feedforward, )
        self.linear2 = nn.Linear(dim_feedforward, input_dim)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.expand = nn.Conv2d(1, c, kernel_size=1)
        self.pooling = nn.AvgPool2d(kernel_size=3, padding=1, stride=1)
        self.proj = nn.Linear(c, input_dim)

        self.activation = F.relu

    def forward(self, src):
        src2, attnScore = self.self_attn(src, src, src, )
        attnScore = attnScore[:, np.newaxis]

        edge_emb = self.expand(attnScore)
        # edge_emb = self.pooling(edge_emb)
        w = edge_emb
        w = w.softmax(dim=-1)
        w = (w * edge_emb).sum(-1).transpose(-1, -2)
        w = self.proj(w)
        src2 = src2 + w

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class intraAFL(nn.Module):
    def __init__(self, input_dim, c):
        super(intraAFL, self).__init__()
        self.input_dim = input_dim
        self.num_block = args.NO_IntraAFL
        NO_head = args.NO_head
        dropout = args.dropout

        self.blocks = nn.ModuleList(
            [intraAFL_Block(input_dim=input_dim, nhead=NO_head, c=c, dropout=dropout) for _ in range(self.num_block)])

        self.fc = DeepFc(input_dim, input_dim)

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        out = out.squeeze()
        out = self.fc(out)
        return out

class Encoder(nn.Module):
    def __init__(self, poi_dim, landUse_dim, input_dim, c):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.densePOI2 = nn.Linear(poi_dim, input_dim)
        self.denseLandUse3 = nn.Linear(landUse_dim, input_dim)

        self.encoderPOI = intraAFL(input_dim, c)
        self.encoderLandUse = intraAFL(input_dim, c)
        self.encoderMob = intraAFL(input_dim, c)
        
        self.activation = F.relu
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        poi_emb, landUse_emb, mob_emb = x

        poi_emb = self.dropout(self.activation(self.densePOI2(poi_emb)))                # (1, 180, 180)
        landUse_emb = self.dropout(self.activation(self.denseLandUse3(landUse_emb)))    # (1, 180, 180)

        poi_emb = self.encoderPOI(poi_emb)                          # (180, 180)
        landUse_emb = self.encoderLandUse(landUse_emb)              # (180, 180)
        mob_emb = self.encoderMob(mob_emb)                          # (180, 180)

        intra_view_embs = [poi_emb, landUse_emb, mob_emb]    # (180, 180)

        return intra_view_embs