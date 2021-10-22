import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import math
from math import sqrt
from math import log
from utils.masking import TriangularCausalMask, ProbMask

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class LogSparceAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(LogSparceAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def log_mask(self, win_len, sub_len):
        mask = torch.zeros((win_len, win_len), dtype=torch.float)
        for i in range(win_len):
            mask[i] = self.row_mask(i, sub_len, win_len)
        return mask.view(1, 1, mask.size(0), mask.size(1))

    def row_mask(self, index, sub_len, win_len):
        log_l = math.ceil(np.log2(sub_len))

        mask = torch.zeros((win_len), dtype=torch.float)
        if((win_len // sub_len) * 2 * (log_l) > index):
            mask[:(index + 1)] = 1
        else:
            while(index >= 0):
                if((index - log_l + 1) < 0):
                    mask[:index] = 1
                    break
                mask[index - log_l + 1:(index + 1)] = 1  # Local attention
                for i in range(0, log_l):
                    new_index = index - log_l + 1 - 2**i
                    if((index - new_index) <= sub_len and new_index >= 0):
                        mask[new_index] = 1
                index -= sub_len
        return mask

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        # if self.mask_flag:
            # if attn_mask is None:
            #     attn_mask = TriangularCausalMask(B, L, device=queries.device)
            # scores.masked_fill_(attn_mask.mask, -np.inf)
        mask = self.log_mask(L, S)
        mask_tri = mask[:, :, :scores.size(-2), :scores.size(-1)]
        scores = scores.to(queries.device)
        mask_tri = mask_tri.to(queries.device)
        scores = scores * mask_tri + -1e9 * (1 - mask_tri)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.contiguous(), attn

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn





# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import einsum
# import numpy as np
# import math
# from math import sqrt,log
# from utils.masking import TriangularCausalMask, ProbMask

# class FullAttention(nn.Module):
#     def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1,heads=8):
#         super(FullAttention, self).__init__()
#         self.scale = scale
#         self.mask_flag = mask_flag
#         self.dropout = nn.Dropout(attention_dropout)

#         self.pre_softmax_proj = nn.Parameter(torch.randn(heads, heads))
#         self.post_softmax_proj = nn.Parameter(torch.randn(heads, heads))    

#     def forward(self, queries, keys, values, attn_mask):
#         B, L, H, E = queries.shape
#         _, S, _, D = values.shape
#         scale = self.scale or 1./sqrt(E)

#         scores = torch.einsum("blhe,bshe->bhls", queries, keys)
    
#         # Talking-Heads Attentio
#         scores = einsum('b h i j, h k -> b k i j', scores, self.post_softmax_proj).contiguous()

#         if self.mask_flag:
#             if attn_mask is None:
#                 attn_mask = TriangularCausalMask(B, L, device=queries.device)

#             scores.masked_fill_(attn_mask.mask, -np.inf)

#         A = self.dropout(torch.softmax(scale * scores, dim=-1))
#         V = torch.einsum("bhls,bshd->blhd", A, values)

#         return V.contiguous()

# class LogSparceAttention(nn.Module):
#     def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
#         super(LogSparceAttention, self).__init__()
#         self.scale = scale
#         self.mask_flag = mask_flag
#         self.output_attention = output_attention
#         self.dropout = nn.Dropout(attention_dropout)

#     def log_mask(self, win_len, sub_len):
#         mask = torch.zeros((win_len, win_len), dtype=torch.float)
#         for i in range(win_len):
#             mask[i] = self.row_mask(i, sub_len, win_len)
#         return mask.view(1, 1, mask.size(0), mask.size(1))

#     def row_mask(self, index, sub_len, win_len):
#         log_l = math.ceil(np.log2(sub_len))

#         mask = torch.zeros((win_len), dtype=torch.float)
#         if((win_len // sub_len) * 2 * (log_l) > index):
#             mask[:(index + 1)] = 1
#         else:
#             while(index >= 0):
#                 if((index - log_l + 1) < 0):
#                     mask[:index] = 1
#                     break
#                 mask[index - log_l + 1:(index + 1)] = 1  # Local attention
#                 for i in range(0, log_l):
#                     new_index = index - log_l + 1 - 2**i
#                     if((index - new_index) <= sub_len and new_index >= 0):
#                         mask[new_index] = 1
#                 index -= sub_len
#         return mask

#     def forward(self, queries, keys, values, attn_mask):
#         B, L, H, E = queries.shape
#         _, S, _, D = values.shape
#         scale = self.scale or 1./sqrt(E)

#         scores = torch.einsum("blhe,bshe->bhls", queries, keys)
#         # if self.mask_flag:
#             # if attn_mask is None:
#             #     attn_mask = TriangularCausalMask(B, L, device=queries.device)
#             # scores.masked_fill_(attn_mask.mask, -np.inf)
#         mask = self.log_mask(L, S)
#         mask_tri = mask[:, :, :scores.size(-2), :scores.size(-1)]
#         scores = scores.to(queries.device)
#         mask_tri = mask_tri.to(queries.device)
#         scores = scores * mask_tri + -1e9 * (1 - mask_tri)

#         A = self.dropout(torch.softmax(scale * scores, dim=-1))
#         V = torch.einsum("bhls,bshd->blhd", A, values)

#         if self.output_attention:
#             return (V.contiguous(), A)
#         else:
#             return (V.contiguous(), None)

# class ProbAttention(nn.Module):
#     def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, heads=8):
#         super(ProbAttention, self).__init__()
#         self.factor = factor
#         self.scale = scale
#         self.mask_flag = mask_flag
#         self.dropout = nn.Dropout(attention_dropout)
       
        
#         self.pre_softmax_proj = nn.Parameter(torch.randn(heads, heads))
#         self.post_softmax_proj = nn.Parameter(torch.randn(heads, heads))
       
#     def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
#         # Q [B, H, L, D]
#         B, H, L_K, E = K.shape
#         _, _, L_Q, _ = Q.shape

#         # calculate the sampled Q_K（计算sample_k=U_part个sampled Q_K）
#         K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
#         index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
#         #index_sample = [96, 25]
#         K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
#         #K_sample = [32, 8, 96, 25, 64]
#         Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()
#         #倒数第二维增加一个维度1，之后删除增加的维度[32, 8, 96, 25]

#         # find the Top_k query with sparisty measurement
#         M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)#计算M[32, 8, 96]
#         M_top = M.topk(n_top, sorted=False)[1]#[32, 8, 25] 第一个25

#         # use the reduced Q to calculate Q_K
#         Q_reduce = Q[torch.arange(B)[:, None, None],
#                      torch.arange(H)[None, :, None],
#                      M_top, :] # factor*ln(L_q)
#         Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

#         return Q_K, M_top

#     def _get_initial_context(self, V, L_Q):
#         B, H, L_V, D = V.shape #V.shape=torch.Size([32, 8, 96, 64]
#         if not self.mask_flag:
#             # V_sum = V.sum(dim=-2)
#             V_sum = V.mean(dim=-2)#[32, 8, 64]
#             contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()#[32, 8, 96, 64]
#         else: # use mask
#             assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
#             contex = V.cumsum(dim=-2)
#         return contex

#     def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
#         B, H, L_V, D = V.shape

#         if self.mask_flag:
#             attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
#             scores.masked_fill_(attn_mask.mask, -np.inf)

#         attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

#         context_in[torch.arange(B)[:, None, None],
#                    torch.arange(H)[None, :, None],
#                    index, :] = torch.matmul(attn, V).type_as(context_in)

#         return context_in

#     def forward(self, queries, keys, values, attn_mask):
#         B, L_Q, H, D = queries.shape#[32, 96, 8, 64]
#         _, L_K, _, _ = keys.shape

#         queries = queries.transpose(2,1)#[32, 8, 96, 64]
#         keys = keys.transpose(2,1)
#         values = values.transpose(2,1)

#         # U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
#         # u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 
        
#         U_part = self.factor * torch.ceil(torch.log(torch.tensor(L_K).float())).int() # c*ln(L_k)
#         u = self.factor * (torch.ceil(torch.log(torch.tensor(L_Q).float())).int()) # c*ln(L_q)
        
#         U_part = U_part if U_part<L_K else L_K
#         u = u if u<L_Q else L_Q
        
#         scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 
        
#         # Talking-Heads Attentio
#         scores_top = einsum('b h i j, h k -> b k i j', scores_top, self.post_softmax_proj).contiguous()
        
#         # add scale factor
#         scale = self.scale or 1./sqrt(D)
#         if scale is not None:
#             scores_top = scores_top * scale
#         # get the context
#         context = self._get_initial_context(values, L_Q)
#         # update the context with selected top_k queries
#         context = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
#         return context.transpose(2,1).contiguous()


# class AttentionLayer(nn.Module):
#     def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
#         super(AttentionLayer, self).__init__()

#         d_keys = d_keys or (d_model//n_heads)
#         d_values = d_values or (d_model//n_heads)

#         self.inner_attention = attention
#         self.query_projection = nn.Linear(d_model, d_keys * n_heads)
#         self.key_projection = nn.Linear(d_model, d_keys * n_heads)
#         self.value_projection = nn.Linear(d_model, d_values * n_heads)
#         self.out_projection = nn.Linear(d_values * n_heads, d_model)
#         self.n_heads = n_heads

#     def forward(self, queries, keys, values, attn_mask):
#         B, L, _ = queries.shape
#         _, S, _ = keys.shape
#         H = self.n_heads
#         #8个头 queries.shape = torch.Size([32, 96, 8, 64])
#         queries = self.query_projection(queries).view(B, L, H, -1)
#         keys = self.key_projection(keys).view(B, S, H, -1)
#         values = self.value_projection(values).view(B, S, H, -1)

#         out = self.inner_attention(
#             queries,
#             keys,
#             values,
#             attn_mask
#         )
#         out = out.view(B, L, -1)

#         return self.out_projection(out)
