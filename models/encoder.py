import torch
import torch.nn as nn
import torch.nn.functional as F

# class ConvLayer(nn.Module):
#     def __init__(self, c_in):
#         super(ConvLayer, self).__init__()
#         self.downConv = nn.Conv1d(in_channels=c_in,
#                                   out_channels=c_in,
#                                   kernel_size=3,
#                                   padding=2,
#                                   padding_mode='circular')
#         self.norm = nn.BatchNorm1d(c_in)
#         self.activation = nn.ELU()
#         self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

#     def forward(self, x):
#         x = self.downConv(x.permute(0, 2, 1))
#         x = self.norm(x)
#         x = self.activation(x)
#         x = self.maxPool(x)
#         x = x.transpose(1,2)
#         return x

class ConvLayer(nn.Module):
    def __init__(self, c_in, d=2):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=0,
                                  stride=1
                                  )
        self.pad1 = nn.Conv1d(in_channels=c_in,
                              out_channels=c_in,
                              kernel_size=1,
                              padding=0,
                              stride=1
                              )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.d = d
        self.dropout = nn.Dropout(0.1)
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # print(self.d)
        if self.d == 1:
            x_i = x.clone()
            x_p1 = self.downConv(x.permute(0, 2, 1))
            x_p2 = self.pad1(x_i[:, 0:2, :].permute(0, 2, 1))
            x_p = torch.cat((x_p1, x_p2), 2)
            x = self.norm(x_p)
            x = self.dropout(self.activation(x))
            x = x + x_i.permute(0, 2, 1)
            x = self.maxPool(x)
            x = x.transpose(1, 2)
            return x
        elif self.d == 2:
            x_i = x.clone()
            x_p = x.permute(0, 2, 1)
            x1 = x[:, 0::2, :]
            x1_p1 = self.downConv(x1.permute(0, 2, 1))
            x1_p2 = self.pad1(x1[:, 0:2, :].permute(0, 2, 1))
            x1_p = torch.cat((x1_p1, x1_p2), 2)
            x2 = x[:, 1::2, :]
            x2_p1 = self.downConv(x2.permute(0, 2, 1))
            x2_p2 = self.pad1(x2[:, 0:2, :].permute(0, 2, 1))
            x2_p = torch.cat((x2_p1, x2_p2), 2)
            for i in range(x_p.shape[2]):
                if i % 2 == 0:
                    x_p[:, :, i] = x1_p[:, :, i // 2]
                else:
                    x_p[:, :, i] = x2_p[:, :, i // 2]
            x = self.norm(x_p)
            x = self.dropout(self.activation(x))
            x = x + x_i.permute(0, 2, 1)
            x = self.maxPool(x)
            x = x.transpose(1, 2)
            return x
        else:
            x_i = x.clone()
            x_p = x.permute(0, 2, 1)
            for i in range(self.d):
                x1 = x[:, i::self.d,:]
                x1_p1 = self.downConv(x1.permute(0, 2, 1))
                x1_p2 = self.pad1(x1[:, 0:2, :].permute(0, 2, 1))
                x1_p = torch.cat((x1_p1, x1_p2), 2)
                for j in range(x_p.shape[2]):
                    if j % self.d == i:
                        x_p[:, :, j] = x1_p[:, :, j // self.d]
            x = self.norm(x_p)
            x = self.dropout(self.activation(x))
            x = x + x_i.permute(0, 2, 1)
            x = self.maxPool(x)
            x = x.transpose(1, 2)
            return x

class lstm(nn.Module):
    def __init__(self,	input_size,	hidden_size,	output_size=1,	num_layers=2):
                    super(lstm,	self).__init__()
                    
                    self.rnn	=	nn.LSTM(input_size,	hidden_size,	num_layers)	
                    self.reg	=	nn.Linear(hidden_size,	output_size)	
                    
    def forward(self,	x):
                    x,	_	=	self.rnn(x)
                    s,	b,	h	=	x.shape
                    x	=	x.view(s*b,	h)
                    x	=	self.reg(x)
                    x	=	x.view(s,	b,	-1)
                    return x

class EncoderLayer(nn.Module):
    def __init__(self,  d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.lstm = lstm(input_size=d_model, hidden_size=d_model, output_size=d_model, num_layers=2)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        new_x= self.lstm(x)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm2(x+y)

class Encoder(nn.Module):
    def __init__(self,lstm,conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.lstm = nn.ModuleList(lstm)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D][32, 96, 512]

        if self.conv_layers is not None:
            for lstm_layer, conv_layer in zip(self.lstm, self.conv_layers):
                x = lstm_layer(x)#x=[32, 96, 512]
                x = conv_layer(x)#x=[32, 49, 512]
              
            x = self.lstm[-1](x)
          
        else:
            for lstm_layer in self.lstm:
                x = lstm_layer(x)


        if self.norm is not None:
            x = self.norm(x)

        return x

