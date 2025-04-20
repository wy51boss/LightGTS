
__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from .layers.pos_encoding import *
from .layers.basics import *
from .layers.attention import *
from src.models.layers.decoder_cnn import Decoder
from .layers.resample_emb import resample_patchemb

import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

            
# Cell
class LightGTS(nn.Module):
    """
    Output dimension: 
         [bs x target_dim x nvars] for prediction
         [bs x target_dim] for regression
         [bs x target_dim] for classification
         [bs x num_patch x n_vars x patch_len] for pretrain
    """
    def __init__(self, c_in:int, target_dim:int, patch_len:int, stride:int, num_patch:int, mask_mode:str = 'patch',mask_nums:int = 3,
                 e_layers:int=3, d_layers:int=3, d_model=128, n_heads=16, shared_embedding=True, d_ff:int=256, 
                 norm:str='BatchNorm', attn_dropout:float=0.4, dropout:float=0., act:str="gelu", 
                 res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='sincos', learn_pe:bool=False, head_dropout = 0, 
                 head_type = "prediction", individual = False, 
                 y_range:Optional[tuple]=None, verbose:bool=False, **kwargs):

        super().__init__()
        assert head_type in ['pretrain', 'prediction', 'regression', 'classification'], 'head type should be either pretrain, prediction, or regression'

        # Basic
        self.num_patch = num_patch
        self.target_dim=target_dim
        self.out_patch_num = math.ceil(target_dim / patch_len)
        self.target_patch_len = 48
        # Embedding
        self.embedding = nn.Linear(self.target_patch_len, d_model, bias=False)
        # self.decoder_embedding = nn.Parameter(torch.randn(1, 1,1, d_model),requires_grad=True)
        self.cls_embedding = nn.Parameter(torch.randn(1, 1, 1, d_model),requires_grad=True)
        # self.sep_embedding = nn.Parameter(torch.randn(1, 1, 1, d_model),requires_grad=True)

        # Position Embedding
        # self.pos = positional_encoding(pe, learn_pe, 1 + num_patch + self.out_patch_num, d_model)
        # self.drop_out = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=e_layers, 
                                    store_attn=store_attn)
        
        # Decoder
        self.decoder = Decoder(d_layers, patch_len=patch_len, d_model=d_model, n_heads=n_heads, d_ff=d_ff,attn_dropout= attn_dropout, dropout=dropout)

        # Head
        self.n_vars = c_in
        self.head_type = head_type
        self.mask_mode = mask_mode
        self.mask_nums = mask_nums
        self.d_model  = d_model
        self.patch_len = patch_len
        



        if head_type == "pretrain":
            self.head = PretrainHead(d_model, patch_len, head_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == "prediction":
            self.head = decoder_PredictHead(d_model, self.patch_len, self.target_patch_len, 9 ,head_dropout)

        # self.apply(self._init_weights)
    
    # def get_dynamic_weights(self, n_preds):
    #     """
    #     Generate dynamic weights for the replicated tokens. This example uses a linearly decreasing weight.
    #     You can modify this to use other schemes like exponential decay, sine/cosine, etc.
    #     """
    #     # Linearly decreasing weights from 1.0 to 0.5 (as an example)
    #     weights = torch.linspace(1.0, 0.5, n_preds)
    #     return weights
    
    def get_dynamic_weights(self, n_preds, decay_rate=0.5):
        """
        Generate dynamic weights for the replicated tokens using an exponential decay scheme.
        
        Args:
        - n_preds (int): Number of predictions to generate weights for.
        - decay_rate (float): The base of the exponential decay. Lower values decay faster (default: 0.9).
        
        Returns:
        - torch.Tensor: A tensor of weights with exponential decay.
        """
        # Exponential decay weights
        weights = decay_rate ** torch.arange(n_preds)
        return weights

    def decoder_predict(self, bs, n_vars, dec_cross):
        """
        dec_cross: tensor [bs x  n_vars x num_patch x d_model]
        """
        # dec_in = self.decoder_embedding.e xpand(bs, self.n_vars, self.out_patch_num, -1)
        # dec_in = self.embedding(self.decoder_len).expand(bs, -1, -1, -1)
        # dec_in = self.decoder_embedding.expand(bs, n_vars, self.out_patch_num, -1)
        # dec_in = dec_cross.mean(2).unsqueeze(2).expand(-1,-1,self.out_patch_num,-1)
        dec_in = dec_cross[:,:,-1,:].unsqueeze(2).expand(-1,-1,2,-1)
        # dec_in = torch.ones_like(dec_in)
        weights = self.get_dynamic_weights(2).to(dec_in.device)
        dec_in = dec_in * weights.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        # dec_in = torch.cat((dec_in, self.sep_tokens), dim=2)
        
        # dec_in = dec_cross[:,:,-self.out_patch_num:,:]
        # dec_in = torch.ones([bs, n_vars, self.out_patch_num, self.d_model]).to(dec_cross.device)
        # dec_in = dec_in + self.pos[-self.out_patch_num:,:]
        decoder_output = self.decoder(dec_in, dec_cross)
        decoder_output = decoder_output.transpose(2,3)

        return decoder_output
        

    def forward(self, z):                             
        """
        z: tensor [bs x num_patch x n_vars x patch_len]
        """   
        bs, num_patch, n_vars, patch_len = z.shape
        
        # tokenizer
        cls_tokens = self.cls_embedding.expand(bs, n_vars, -1, -1)
        
        embedding = nn.Linear(patch_len, self.d_model, bias=False)
        embedding.weight.data = resample_patchemb(old=self.embedding.weight.data, new_patch_len=self.patch_len)
        # visualize_embedding_weights(self.embedding.weight.data, save_path="embedding_heatmap.pdf", cmap='Reds')

        z = embedding(z).permute(0,2,1,3) # [bs x n_vars x num_patch x d_model]
        

        token_embedding = z.mean(2)[0,-1,:]
        token_embedding_numpy = token_embedding.cpu().detach().numpy()
        df = pd.DataFrame([token_embedding_numpy])
        df.to_csv("embedding_exp/etth1_token_embedding_period.csv", index=False, header=False)


        z = torch.cat((cls_tokens, z), dim=2)  # [bs x n_vars x (1 + num_patch) x d_model]

        # z = self.drop_out(z + self.pos[:1 + self.num_patch, :])

        # encoder 
        z = torch.reshape(z, (-1, 1 + num_patch, self.d_model)) # [bs*n_vars x num_patch x d_model]
        z = self.encoder(z)
        z = torch.reshape(z, (-1, n_vars, 1 + num_patch, self.d_model)) # [bs, n_vars x num_patch x d_model]

        # plot_cosine_similarity_heatmap(z[:,:,1:,:], 'embedding_exp/weather_f_s.pdf')


        # df.to_csv("ETTh1_token_embedding_period.csv", index=False, header=False)

        # decoder
        z = self.decoder_predict(bs, n_vars, z[:,:,:,:])
        
        # predict
        z = self.head(z[:,:,:,:])    
        z = z[:,:self.target_dim, :, [3,4,5]]
        z = z[:,:self.target_dim, :, :].mean(3)


        # z: [bs x target_dim x nvars] for prediction
        #    [bs x target_dim] for regression
        #    [bs x target_dim] for classification
        #    [bs x num_patch x n_vars x patch_len] for pretrain
        return z


class PretrainHead(nn.Module):
    def __init__(self, d_model, patch_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x):
        """
        x: tensor [bs x nvars x d_model x num_patch]
        output: tensor [bs x nvars x num_patch x patch_len]
        """

        x = x.transpose(2,3)                     # [bs x nvars x num_patch x d_model]
        x = self.linear( self.dropout(x) )      # [bs x nvars x num_patch x patch_len]
        x = x.permute(0,2,1,3)                  # [bs x num_patch x nvars x patch_len]
        return x


class decoder_PredictHead(nn.Module):
    def __init__(self, d_model, patch_len, target_patch_len, num_classes, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.patch_len = patch_len
        self.d_model = d_model
        self.num_classes = num_classes
        
        # Separate linear layers for each class
        self.linears = nn.ModuleList([nn.Linear(d_model, target_patch_len, bias=False) for _ in range(num_classes)])

    def forward(self, x):
        """
        x: tensor [bs x nvars x d_model x num_patch]
        output: tensor [bs x num_patch x patch_len x nvars x num_classes]
        """
        x = x.transpose(2, 3)  # [bs x nvars x num_patch x d_model]
        outputs = []

        # Apply each linear layer to the input for each class
        for linear in self.linears:
            linear.weight.data = resample_patchemb(old=linear.weight.data.T, new_patch_len=self.patch_len).T
            output = linear(self.dropout(x)).unsqueeze(-1)  # [bs x nvars x num_patch x patch_len x 1]
            outputs.append(output)

        x = torch.cat(outputs, dim=-1)  # [bs x nvars x num_patch x patch_len x num_classes]
        x = x.permute(0, 2, 3, 1, 4)  # [bs x num_patch x patch_len x nvars x num_classes]
        return x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])


class PatchTSTEncoder(nn.Module):
    def __init__(self, c_in, num_patch, patch_len, n_embedding=32,
                 n_layers=3, d_model=128, n_heads=16, shared_embedding=True,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):

        super().__init__()
        self.n_vars = c_in
        self.num_patch = num_patch
        self.patch_len = patch_len
        self.d_model = d_model
        self.shared_embedding = shared_embedding
        self.n_embedding=n_embedding         

        # Input encoding: projection of feature vectors onto a d-dim vector space
        if not shared_embedding: 
            self.W_P = nn.ModuleList()
            for _ in range(self.n_vars): self.W_P.append(nn.Linear(patch_len, d_model))
        else:
            self.W_P = nn.Linear(patch_len, d_model)     


        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, num_patch, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, 
                                    store_attn=store_attn)

    def forward(self, x) -> Tensor:       
        """
        x: tensor [bs x num_patch x nvars x patch_len]
        """
        bs, num_patch, n_vars, patch_len = x.shape      # x: [bs x num_patch x nvars x d_model]
        # Input encoding
        if not self.shared_embedding:
            x_out = []
            for i in range(n_vars): 
                z = self.W_P[i](x[:,:,i,:])
                x_out.append(z)
            x1 = torch.stack(x_out, dim=2)
        else:
            x1 = self.W_P(x)                                                    
        x1 = x1.transpose(1,2)
     
        u = torch.reshape(x1, (bs*n_vars, num_patch, self.d_model) )              # u: [bs * nvars x num_patch x d_model]
        u = self.dropout(u + self.W_pos)                                         # u: [bs * nvars x num_patch x d_model]
        
        # Encoder
        z = self.encoder(u)                                                      # z: [bs * nvars x num_patch x d_model]
        z = torch.reshape(z, (-1,n_vars, num_patch, self.d_model))               # z: [bs x nvars x num_patch x d_model]
        z = z.permute(0,1,3,2)                                                 # z: [bs x nvars x d_model x num_patch]

        return z

    
# Cell
class TSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(d_model, n_heads=n_heads, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src:Tensor):
        """
        src: tensor [bs x q_len x d_model]
        """
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores)
            return output
        else:
            for mod in self.layers: output = mod(output)
            return output



class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, store_attn=False,
                 norm='LayerNorm', attn_dropout=0, dropout=0., bias=True, 
                activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

        # # se block
        # self.SE = SE_Block(inchannel=7)


    def forward(self, src:Tensor, prev:Optional[Tensor]=None):
        """
        src: tensor [bs x q_len x d_model]
        """
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev)
        else:
            # attention_mask = causal_attention_mask(src.shape[1]).to(src.device)
            # src2, attn = self.self_attn(src, src, src, attn_mask=attention_mask)
            src2, attn = self.self_attn(src, src, src)
        if self.store_attn:
            self.attn = attn
        
        # total, num_patch, d_model = src2.size()
        # bs = int(total/7)

        # src2 = self.SE(src2.reshape(bs, 7, num_patch, -1)).reshape(total, num_patch, -1)


        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src


def causal_attention_mask(seq_length):
    """
    创建一个因果注意力掩码。掩码中的每个位置 (i, j) 
    表示在计算第i个位置的attention时, 第j个位置是否可以被看见。
    如果j <= i, 这个位置被设为1(可见), 否则设为0(不可见)。
    
    Args:
        seq_length (int): 序列的长度
    
    Returns:
        torch.Tensor: 因果注意力掩码，大小为 (seq_length, seq_length)
    """
    mask = torch.triu(torch.ones(seq_length, seq_length) * float('-inf'), diagonal=1)
    return mask



    """
    Calculate and save the cosine similarity heatmap for patches in the first batch,
    considering the last dimension of n_vars.

    Parameters:
    z (torch.Tensor): Input tensor of shape [batch_size, n_vars, num_patch, d_model].
    save_path (str): Path to save the heatmap image.
    """
    # Permute the tensor to [batch_size, num_patch, n_vars, d_model]
    z = z.permute(0, 2, 1, 3)  # Ensure correct shape

    # Take the first batch: [num_patch, n_vars, d_model]
    z_first_batch = z[0]

    # Take the last dimension of n_vars: [num_patch, d_model]
    z_last_dim = z_first_batch[:, -1, :]

    # Group the patches into chunks of 6 and average them
    patch_group_size = 2
    num_patches = z_last_dim.shape[0]
    
    if num_patches % patch_group_size != 0:
        raise ValueError(f"The number of patches ({num_patches}) is not a multiple of {patch_group_size}.")

    # Reshape and average every 6 patches: [num_patch // patch_group_size, patch_group_size, d_model]
    z_grouped_avg = z_last_dim.view(num_patches // patch_group_size, patch_group_size, -1).mean(dim=1)

    # Normalize the patches for cosine similarity
    z_norm = z_grouped_avg / torch.norm(z_grouped_avg, dim=1, keepdim=True)

    # Compute cosine similarity: [num_patch, num_patch]
    cosine_sim = torch.matmul(z_norm, z_norm.T)

    # Convert to numpy for visualization
    cosine_sim_np = cosine_sim.cpu().numpy()

    # Calculate the number of blocks (each of size 36)
    block_size = 1
    num_patches = cosine_sim_np.shape[0]

    # Ensure that the number of patches is a multiple of block_size
    if num_patches % block_size != 0:
        raise ValueError(f"The number of patches ({num_patches}) is not a multiple of {block_size}.")

    # Reshape the cosine similarity matrix into blocks of 36 patches
    cosine_sim_avg = cosine_sim_np.reshape(num_patches // block_size, block_size, num_patches // block_size, block_size)

    # Average within each block: [num_blocks, block_size, num_blocks, block_size]
    cosine_sim_avg = cosine_sim_avg.mean(axis=(1, 3))

    # Plot heatmap using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cosine_sim_avg, cmap="coolwarm", annot=False, cbar=True, vmax=1, vmin=0)
    plt.title("Cosine Similarity Heatmap Between Patches")
    plt.xlabel("Patch Index")
    plt.ylabel("Patch Index")
    
    # Save the heatmap to the specified path
    plt.savefig(save_path)
    plt.close()