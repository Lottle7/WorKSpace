import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, embedding_size, hidden_dim):
        super(Decoder, self).__init__()
        self.decoder_s = nn.Linear(embedding_size, embedding_size)
        self.decoder_t = nn.Linear(embedding_size, embedding_size)

        self.decoder_n = nn.Linear(embedding_size, embedding_size)  
        self.decoder_p = nn.Linear(embedding_size, embedding_size)  
        self.decoder_l = nn.Linear(embedding_size, embedding_size)
        self.decoder_satteliteImage = nn.Linear(hidden_dim, 1) 

    def forward(self, x, ablation=None, si_emb=None):
        out_p = self.decoder_p(x)
        out_l = self.decoder_l(x)

        if ablation is None or "attn" in ablation:
            out_s = self.decoder_s(x)
            out_t = self.decoder_t(x)
        if ablation is not None and "gnn" in ablation:
            out_n = self.decoder_n(x)
        if ablation is not None and si_emb is not None:
            out_satteliteImage = self.decoder_satteliteImage(si_emb)

        if ablation is None:    # attn
            return out_s, out_t, out_p, out_l
        elif si_emb is not None and "attn" in ablation:    # attn+cnn
            return out_s, out_t, out_p, out_l, out_satteliteImage
        elif si_emb is None and "gnn" in ablation:    # gnn
            return out_p, out_l, out_n
        elif si_emb is not None and "gnn" in ablation:    # gnn+cnn
            return out_p, out_l, out_n, out_satteliteImage
