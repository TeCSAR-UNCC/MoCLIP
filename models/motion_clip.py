import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QuickGELU(nn.Module):
    """GELU activation function used in CLIP."""
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    """Single Transformer Block: Multi-Head Attention + MLP + LayerNorm."""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)  # Batch-first for easier processing
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            QuickGELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )
        self.ln_2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x))[0]  # Self-Attention
        x = x + self.mlp(self.ln_2(x))  # MLP
        return x
        
        
class TemporalAttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attention = nn.Linear(embed_dim, 1)  # Scalar score per timestep

    def forward(self, x):
        # x: (batch, temporal, embed_dim) → (batch, temporal, 1)
        weights = torch.softmax(self.attention(x), dim=1)  # (128, 196, 1)
        x = (x * weights).sum(dim=1)  # Weighted sum → (128, 768)
        return x
    
class PositionalEncoding(nn.Module):
    #Borrow from MDM, the same as above, but add dropout, exponential may improve precision
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) #[max_len, 1, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)
        
    
n_joint  = 22

class MotionEncoder(nn.Module):
    def __init__(self, dataset_name, hidden_size=256, dropout=0.1, hidden_dim=128,device=None):
        super().__init__()
        # input_dim ex: vocab_size
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size

        if dataset_name == 'kit':
            #KIT-ML GRAPH
            [root, torso, lfarm, rram, lfleg, rleg, hands , feet] = [0], [1, 2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16,17,18,19,20], [9, 10, 7, 6], [17, 20, 15, 12]
            self.body_parts = [root, torso, lfarm, rram, lfleg, rleg, hands, feet]
            njoint_per_part = [len(p) for p in [root, torso, lfarm, rram, lfleg, rleg, hands, feet]]

        elif dataset_name == 't2m':
            # HUMAN ML3D GRAPH
            [root, torso, rarm, larm, rleg, lleg, hands, feet] = [0], [0, 3, 6, 9, 12, 15], [14, 17, 19, 21], [ 13, 16, 18, 20], [2, 5, 8, 11], [1, 4, 7, 10], [19, 21, 18, 20], [8, 11, 18, 20]
            self.body_parts = [root, torso, rarm, larm, rleg, lleg, hands, feet]
            njoint_per_part = [len(p) for p in self.body_parts]
        else:
            raise KeyError('Dataset Does Not Exist')
        
        self.six_layers = nn.ModuleList([nn.Linear(inp_dim*3,hidden_dim,device=device) for inp_dim in njoint_per_part])
        self.six_layers_2 = nn.ModuleList([nn.Linear(hidden_dim,hidden_size//2,device=device) for _ in njoint_per_part])
        self.six_layers_v = nn.ModuleList([nn.Linear(inp_dim*3,hidden_dim,device=device) for inp_dim in njoint_per_part])
        self.six_layers_v2 = nn.ModuleList([nn.Linear(hidden_dim,hidden_size//2,device=device) for _ in njoint_per_part])

    def forward(self, x):
        N, T, n_joint, dim = x.size()
        x = x.transpose(1, 0)

        skel_parts = [x[:, :, prt, :].reshape(T,N, len(prt) * 3) for prt in self.body_parts]

        velocity = torch.cat([(x[0] - x[0]).unsqueeze(0),  x[1:]-x[:-1]], dim=0)
        skel_parts_v = [velocity[:, :, prt, :].reshape(T,N, len(prt) * 3) for prt in self.body_parts]

        # Pose features "P_ij"
        outputs = torch.zeros(x.size()[:-2]+(self.hidden_size,len(self.body_parts)))
        i = 0
        for prt,prt_v in zip(skel_parts,skel_parts_v):
            out = torch.tanh(self.six_layers[i](prt))
            out = self.dropout(out)
            out = torch.tanh(self.six_layers_2[i](out))
            out_v = torch.tanh(self.six_layers_v[i](prt_v))
            out_v = self.dropout(out_v)
            out_v = torch.tanh(self.six_layers_v2[i](out_v))
            outputs[:,:,:,i] = torch.cat([out,out_v],dim=-1)
            i += 1

        outputs = outputs.flatten(2).transpose(1, 0)

        return outputs

class MotionTransformer(nn.Module):
    """Transformer Model for Motion Data (Temporal-Spatial Attention)."""
    def __init__(self, dataset_name, temporal_dim=196, spatial_dim=256, embed_dim=1032, num_heads=12, num_layers=12, output_dim=512, dropout=0.1, device="cuda"):
        super().__init__()

        self.device = device
        self.dropout = dropout
        # Linear projection instead of convolution (since motion data isn't an image)
        self.encoder = MotionEncoder(dataset_name, hidden_size=spatial_dim)
        self.embed = nn.Linear(spatial_dim*len(self.encoder.body_parts), embed_dim)  # (263 → 768)
        self.position_enc = PositionalEncoding(embed_dim, self.dropout)

        self.ln_pre = nn.LayerNorm(embed_dim)
        self.transformer = nn.Sequential(*[ResidualAttentionBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.ln_post = nn.LayerNorm(embed_dim)

        self.temporal_reduction = TemporalAttentionPooling(embed_dim)
        self.proj = nn.Linear(embed_dim, output_dim)  # Project to 512-d for CLIP compatibility

    def forward(self, x):
        """
        Input Shape: (batch_size, temporal_dim, spatial_dim) = (256, 196, 263)
        Output Shape: (batch_size, 512)
        """
        # batch_size, temporal_dim, spatial_dim = x.shape

        x = self.encoder(x).to(self.device)
        x = self.embed(x)  # (128, 196, 256*6) → (128, 196, 768)

        # Transformer Processing
        x = self.ln_pre(x)
        x = self.position_enc(x)
        x = self.transformer(x)
        x = self.ln_post(x) 

        # Create a mask for non-padded entries (True where not padded)
        # non_padded_mask = ~temporal_mask  # Invert mask: True for non-padding
        
        x = torch.mean(x, dim=1)

        # Final projection to CLIP 512-d space
        x = self.proj(x)  # (128, 512)

        return x
