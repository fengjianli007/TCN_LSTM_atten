import torch
import torch.nn as nn
import numpy as np

from math import sqrt
from utils.masking import  ProbMask

class ProbAttention(nn.Module):
    def __init__(self, factor=5, scale=None):
        super(ProbAttention, self).__init__()
        '''
        maskflag= False for encoder, True for decoder
        factor: 5
        scale:None
        dropout:0.05
        output_attention:false
        '''
        self.factor = factor
        self.scale = scale
        # self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E) # 96,8,96,96,64
        #这样会在L_Q维度复制 L_Q个L_K * E 
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q  #在L_K的的范围内，随机生成L_Q * 25的数值 
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :] #32,8,96,25,64
        # 这里先根据index_sample的index对K进行采样
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze() #32,8,96,25
        # 然后计算U个个QK乘积对

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K) #32,8,96
        # print('Mshape',M.shape)
        M_top = M.topk(n_top, sorted=False)[1]  #选出n_top个最重要的Q 的索引  #32 * 8 * 25
        #M_top = M.topk(25, sorted=False)[1] 

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)   #32，8，25，64
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) #计算点积  32，8，25，96
        # factor*ln(L_q)*L_k，用选出的Q和K进行乘积配对选出最有统治力的QK，也就是attention中最重要的部分。

        return Q_K, M_top #返回点积矩阵和索引

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape #32，8，96，64
  
        # V_sum = V.sum(dim=-2)
        V_sum = V.mean(dim=-2)
        # print('V_sum shape',V_sum.shape)
        contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        # print('contex shape',contex.shape)

        return contex

    def _update_context(self, context_in, V, scores, index, L_Q):
        '''_update_context(context, values, scores_top, index, L_Q, attn_mask)'''
        B, H, L_V, D = V.shape
        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores) 32 * 8 * 25 * 96

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V) #32 * 8 * 96 * 64

        return (context_in)

    def forward(self, queries, keys, values):
        B, L_Q, H, D = queries.shape #32,96,8,64
        _, L_K, _, _ = keys.shape # 96
        #print(self.mask_flag)
        queries = queries.view(B, H, L_Q, -1) #32,8,96,64
        keys = keys.view(B, H, L_K, -1) #32,8,96,64
        values = values.view(B, H, L_K, -1) #32,8,96,64

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)   ln(L_k) 向上取整 #25
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q)  #25
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 
        #得到计算好的Q_K乘积矩阵和筛选出来的Q的索引
        # add scale factor
        scale = self.scale or 1./sqrt(D) # 这缩放因子和Transformer中的dk是一样的
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q) #32，8，96，64
        # update the context with selected top_k queries
        context = self._update_context(context, values, scores_top, index, L_Q) # attention过后依旧得到等数量的context
        # print('update context shape',context.shape)
        return context.contiguous() # view过后都需要这样？？？


class AttentionLayer(nn.Module):
    # 汇总的函数，根据使用的超参数进行调整attention的类别，大小等
    def __init__(self, d_model, n_heads, dropout_p=0.1, d_keys=None,
                 d_values=None):
                # '''
                # AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
#d_model, #512
                                    # n_heads #8
                                    # )
                                    # '''
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model//n_heads) #512/8 = 64
        d_values = d_values or (d_model//n_heads) # 64

        self.inner_attention = ProbAttention(factor=5, scale=None)#Attn ProbAttention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads) #512 ， 64*8 = 512
        self.key_projection = nn.Linear(d_model, d_keys * n_heads) # 512 , 64*8 = 512
        self.value_projection = nn.Linear(d_model, d_values * n_heads) #512, 64*8 = 512
        self.out_projection = nn.Linear(d_values * n_heads, d_model) #512,512
    
        self.dropout = nn.Dropout(p=dropout_p)
        self.n_heads = n_heads #8

    def forward(self,inputs):
        '''
        queries,keys,values全部为 embedding 32 * 96 *512
        attn_mask为none

        这里的BLS大小是根据输入大小而调整的
        '''
        
        B, L, _ = inputs.shape 
        _, S, _ = inputs.shape
        H = self.n_heads
        # B = 32 , L = 96 , S = 96 , H = 8
        queries = self.query_projection(inputs).view(B, L, H, -1) #投影 32 * 96 * 512 ——> 32 * 96 * 8 * 64
        keys = self.key_projection(inputs).view(B, S, H, -1)#投影 32 * 96 * 512 ——> 32 * 96 * 8 * 64
        values = self.value_projection(inputs).view(B, S, H, -1)#投影 32 * 96 * 512 ——> 32 * 96 * 8 * 64

        out = self.inner_attention( 
            queries,
            keys,
            values
        )#ProbAttention
        out = out.view(B, L, -1)
        out = self.out_projection(out)
        # return self.out_projection(out), attn
        return out
        # return self.dropout(out)

