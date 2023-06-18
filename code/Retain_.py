# -*-coding:utf-8-*-
import torch.nn as nn
import torch.nn.functional as F
import torch


class Sequence_encoder_lstm(nn.Module):
    def __init__(self, input_dimension, emb_dimension, hidden_dimension, layers=2, bi_direction=False, drop=0.5,
                 data_source='mimic', bias_tag=True):
        super(Sequence_encoder_lstm, self).__init__()
        # self.embedding = nn.Sequential(nn.Linear(in_features=input_dimension, out_features=emb_dimension, bias=True),
        #                                nn.Dropout(drop),
        #                                nn.Linear(in_features=emb_dimension, out_features=emb_dimension, bias=True))
        self.embedding = nn.Sequential(nn.Linear(in_features=input_dimension, out_features=emb_dimension, bias=True),
                                       nn.Dropout(drop))
        # self.embedding = nn.Linear(in_features=input_dimension, out_features=emb_dimension, bias=bias_tag)
        self.encoder = nn.LSTM(input_size=emb_dimension, hidden_size=hidden_dimension, num_layers=layers,
                               batch_first=True, bias=True, bidirectional=bi_direction, dropout=drop)
        self.data_source = data_source
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop)

    def forward(self, x):  # x: b x t x in
        if self.data_source != 'eicu':
            emb = self.embedding(x)  # emb: b x t x emb_dim
            emb_relu = self.relu(emb)
            x, _ = self.encoder(emb_relu)  # x-> b x t x h or 2h
        else:
            x_, mask, length = x
            mask = mask[:, :, 0]
            emb = self.embedding(x_).permute(2, 0, 1) * mask
            emb = emb.permute(1, 2, 0).float()
            emb = self.dropout(emb)
            emb_relu = self.relu(emb)
            length = length.detach().cpu()
            pack_emb = torch.nn.utils.rnn.pack_padded_sequence(emb_relu, length, batch_first=True)
            x, _ = self.encoder(pack_emb)
            x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
            if x.shape[1] != x_.shape[1]:
                pad = torch.zeros((x_.shape[0], x_.shape[1]-x.shape[1], x.shape[-1])).cuda()
                x = torch.cat((x, pad), dim=1)
        return x, emb


class Theta_Encoder(nn.Module):
    def __init__(self, opts):
        super(Theta_Encoder, self).__init__()
        self.model_name = opts.model_name
        bias_tag = False
        if opts.gen_net == 'LSTM':
            self.sequence_encoder = Sequence_encoder_lstm(opts.in_dim, opts.emb_dim, opts.hidden_dim, opts.layers,
                                                          opts.bi_direction, opts.drop, opts.data_source, bias_tag)
        else:
            self.sequence_encoder = Sequence_encoder_lstm(opts.in_dim, opts.emb_dim, opts.hidden_dim, opts.layers,
                                                          opts.bi_direction, opts.drop, opts.data_source, bias_tag)

        if not opts.bi_direction:
            self.alpha_layer = nn.Linear(opts.hidden_dim, out_features=1, bias=True)
            self.beta_layer = nn.Sequential(nn.Linear(opts.hidden_dim, out_features=opts.emb_dim, bias=True),
                                            nn.Tanh())
        else:
            self.alpha_layer = nn.Linear(opts.hidden_dim * 2, out_features=1, bias=True)
            self.beta_layer = nn.Sequential(nn.Linear(opts.hidden_dim * 2, out_features=opts.emb_dim, bias=True),
                                            nn.Tanh())

        self.predictor = nn.Linear(opts.emb_dim, 2, bias=True)
        self.interpret = opts.interpret
        self.data_source = opts.data_source

        # self.leakyrelu = nn.LeakyReLU()

    def forward(self, x, z=None):
        # TODO: x * z    z[:, :, 1] [z, z, z, z, ]
        if self.model_name == 'Retain':
            x, emb = self.sequence_encoder(x)
            alpha = F.softmax(self.alpha_layer(x).transpose(1, 2), dim=-1)  # -> b, 1, t
            beta = self.beta_layer(x)
            context = torch.bmm(alpha, beta * emb).squeeze(1)
            out = self.predictor(context)  # b x emb_dim
            if self.interpret:
                return out, context, alpha, beta
            else:
                return out

        if self.model_name == 'Dipole':
            x, emb = self.sequence_encoder(x)
            alpha = F.softmax(self.alpha_layer(x).transpose(1, 2), dim=-1)
            context = torch.bmm(alpha, emb)
            out = self.predictor(context)
            return out


class Retain(nn.Module):
    def __init__(self, opts):
        super(Retain, self).__init__()
        opts.model_name = 'Retain'
        self.encoder = Theta_Encoder(opts)
        self.interpret = opts.interpret
        self.data_source = opts.data_source

    def forward(self, x):
        # if self.data_source != 'eicu':
        if self.interpret:
            out, context, alpha, beta = self.encoder(x)
            return out, context, alpha, beta
        else:
            out = self.encoder(x)
            return out