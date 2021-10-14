import torch
import torch.nn as nn
import torch.nn.functional as F
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding

from models.Conformer_encoder.conformer_encoder import ConformerEncoder


class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.05, embed='fixed', freq='h', activation='gelu', 
                device: torch.device = 'cuda'):
        super(Informer, self).__init__()
        self.pred_len = out_len
    
        # embedding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)

        # encoder

        self.encoder = ConformerEncoder(
            encoder_dim=d_model,
            num_layers=e_layers,
            n_heads=n_heads,
            feed_forward_expansion_factor=4,
            conv_expansion_factor=2,
            feed_forward_dropout_p=dropout,
            attention_dropout_p=dropout,
            conv_dropout_p=dropout,
            conv_kernel_size=31,
            half_step_residual=True,
            factor=factor,
            device=device,
        )
        # # Decoder 
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(ProbAttention(mask_flag=True, factor=factor, heads=n_heads), 
                                d_model, n_heads),
                    AttentionLayer(FullAttention(mask_flag=False, factor=factor, attention_dropout=dropout,heads=n_heads), 
                                d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)#enc_out[32, 96, 512]
        # enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        enc_out= self.encoder(enc_out)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out) 
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)

        return dec_out[:,-self.pred_len:,:] # [B, L, D]


