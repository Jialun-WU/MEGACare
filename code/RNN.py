import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import time

class RETAIN(nn.Module):
    def __init__(self, dim_input, dim_emb=128, dropout_input=0, dropout_emb=0.6, dim_alpha=128, dim_beta=128,
                 dropout_context=0.6, dim_output=2, l2=0.0001, batch_first=True):
        super(RETAIN, self).__init__()
        self.batch_first = batch_first
        self.embedding = nn.Sequential(
            nn.Dropout(p=dropout_input),
            nn.Linear(dim_input, dim_emb, bias=False),
            nn.Dropout(p=dropout_emb)
        )
        init.xavier_normal(self.embedding[1].weight)

        self.rnn_alpha = nn.GRU(input_size=dim_emb, hidden_size=dim_alpha, num_layers=1, batch_first=self.batch_first)

        self.alpha_fc = nn.Linear(in_features=dim_alpha, out_features=1)
        init.xavier_normal(self.alpha_fc.weight)
        self.alpha_fc.bias.data.zero_()

        self.rnn_beta = nn.GRU(input_size=dim_emb, hidden_size=dim_beta, num_layers=1, batch_first=self.batch_first)

        self.beta_fc = nn.Linear(in_features=dim_beta, out_features=dim_emb)
        init.xavier_normal(self.beta_fc.weight, gain=nn.init.calculate_gain('tanh'))
        self.beta_fc.bias.data.zero_()

        self.output = nn.Sequential(
            nn.Dropout(p=dropout_context),
            nn.Linear(in_features=dim_emb, out_features=dim_output)
        )
        init.xavier_normal(self.output[1].weight)
        self.output[1].bias.data.zero_()

    def forward(self, x, lengths):
        if self.batch_first:
            batch_size, max_len = x.size()[:2]
        else:
            max_len, batch_size = x.size()[:2]

        emb = self.embedding(x)
        packed_input = pack_padded_sequence(emb, lengths, batch_first=self.batch_first)

        g, _ = self.rnn_alpha(packed_input)

        # alpha_unpacked -> batch_size X max_len X dim_alpha
        alpha_unpacked, _ = pad_packed_sequence(g, batch_first=self.batch_first)

        # mask -> batch_size X max_len X 1
        mask = Variable(torch.FloatTensor(
            [[1.0 if i < lengths[idx] else 0.0 for i in range(max_len)] for idx in range(batch_size)]).unsqueeze(2),
                        requires_grad=False)

        if next(self.parameters()).is_cuda:  # returns a boolean
            mask = mask.cuda()
        # e => batch_size X max_len X 1
        e = self.alpha_fc(alpha_unpacked)

        def masked_softmax(batch_tensor, mask):
            exp = torch.exp(batch_tensor)
            masked_exp = exp * mask
            sum_masked_exp = torch.sum(masked_exp, dim=1, keepdim=True)
            return masked_exp / sum_masked_exp

        alpha = masked_softmax(e, mask)

        h, _ = self.rnn_beta(packed_input)

        beta_unpacked, _ = pad_packed_sequence(h, batch_first=self.batch_first)

        beta = F.tanh(self.beta_fc(beta_unpacked) * mask)

        context = torch.bmm(torch.transpose(alpha, 1, 2), beta * emb).squeeze(1)

        # without applying non-linearity
        logit = self.output(context)

        return logit


class RNN(nn.Module):
    def __init__(self, dim_input, dim_emb=128, dropout_input=0.5, dropout_emb=0.6, dim_alpha=128, dim_beta=128,
                 dropout_context=0.6, dim_output=2, l2=0.0001, batch_first=True):
        super(RNN, self).__init__()
        self.batch_first = batch_first
        self.embedding = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(dim_input, dim_emb, bias=False),
            nn.Dropout(p=0.1),
            nn.ReLU()
        )
        init.xavier_normal(self.embedding[1].weight)

        self.rnn_alpha = nn.LSTM(input_size=dim_emb, hidden_size=dim_alpha, num_layers=1, batch_first=self.batch_first)
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features=dim_emb, out_features=dim_output),
            nn.Dropout(p=dropout_context),
        )
        init.xavier_normal(self.output[1].weight)
        self.output[1].bias.data.zero_()

    def forward(self, x, lengths):
        if self.batch_first:
            batch_size, max_len = x.size()[:2]
        else:
            max_len, batch_size = x.size()[:2]

        emb = self.embedding(x)
        packed_input = pack_padded_sequence(emb, lengths, batch_first=self.batch_first)
        g, _ = self.rnn_alpha(packed_input)
        alpha_unpacked, _ = pad_packed_sequence(g, batch_first=self.batch_first)
        lens = emb.size(1)
        # vector = Variable(torch.FloatTensor([[1.0 for i in range(lens)] for idx in range(batch_size)]).unsqueeze(2),
        #                   requires_grad=False)
        # vector = vector.cuda()
        context = torch.sum(alpha_unpacked, dim=1)
        # context = torch.bmm(torch.transpose(vector, 1, 2), alpha_unpacked).squeeze(1)
        logit = self.output(context)
        return logit


class RNN_plus(nn.Module):
    def __init__(self, dim_input, dim_emb=128, dropout_input=0.5, dropout_emb=0.6, dim_alpha=128, dim_beta=128,
                 dropout_context=0.6, dim_output=2, batch_first=True):
        super(RNN_plus, self).__init__()
        self.batch_first = batch_first
        self.embedding = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(dim_input, dim_emb, bias=False),
            nn.Dropout(p=0.2),
            nn.ReLU()
        )
        init.xavier_normal(self.embedding[1].weight)

        self.rnn_alpha = nn.GRU(input_size=dim_emb, hidden_size=dim_alpha, num_layers=2, batch_first=self.batch_first)
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features=dim_emb, out_features=dim_output),
            nn.Dropout(p=dropout_context),
        )
        init.xavier_normal(self.output[1].weight)
        self.output[1].bias.data.zero_()

    def forward(self, x, lengths):
        if self.batch_first:
            batch_size, max_len = x.size()[:2]
        else:
            max_len, batch_size = x.size()[:2]

        emb = self.embedding(x)
        packed_input = pack_padded_sequence(emb, lengths, batch_first=self.batch_first)
        g, _ = self.rnn_alpha(packed_input)
        alpha_unpacked, _ = pad_packed_sequence(g, batch_first=self.batch_first)
        lens = emb.size(1)
        vector = Variable(torch.FloatTensor([[1.0 for i in range(lens)] for idx in range(batch_size)]).unsqueeze(2),
                          requires_grad=False)
        vector = vector.cuda()
        context = torch.bmm(torch.transpose(vector, 1, 2), alpha_unpacked).squeeze(1)
        logit = self.output(context)
        return logit

class DIPLE(nn.Module):
    def __init__(self, dim_input, dim_emb=128, dropout_input=0, dropout_emb=0.6, dim_alpha=128, dim_beta=128,
                 dropout_context=0.6, dim_output=2, l2=0.0001, batch_first=True):
        super(DIPLE, self).__init__()
        self.batch_first = batch_first
        self.embedding = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(dim_input, dim_emb, bias=False),
            nn.Dropout(p=0.5),
            nn.ReLU()
        )
        init.xavier_normal(self.embedding[1].weight)

        self.rnn_alpha = nn.GRU(input_size=dim_emb, hidden_size=dim_alpha/2, num_layers=1,
                                 batch_first=self.batch_first, bidirectional=True)

        self.alpha_fc = nn.Linear(in_features=dim_alpha, out_features=1)
        init.xavier_normal(self.alpha_fc.weight)
        self.alpha_fc.bias.data.zero_()

        self.output = nn.Sequential(
            nn.Linear(in_features=dim_alpha, out_features=dim_output),
            nn.Dropout(p=dropout_context),
        )
        init.xavier_normal(self.output[0].weight)
        self.output[0].bias.data.zero_()

    def forward(self, x, lengths):
        if self.batch_first:
            batch_size, max_len = x.size()[:2]
        else:
            max_len, batch_size = x.size()[:2]

        emb = self.embedding(x)
        packed_input = pack_padded_sequence(emb, lengths, batch_first=self.batch_first)
        g, _ = self.rnn_alpha(packed_input)
        alpha_unpacked, _ = pad_packed_sequence(g, batch_first=self.batch_first)

        mask = Variable(torch.FloatTensor(
            [[1.0 if i < lengths[idx] else 0.0 for i in range(max_len)] for idx in range(batch_size)]).unsqueeze(2),
                        requires_grad=False)

        if next(self.parameters()).is_cuda:  # returns a boolean
            mask = mask.cuda()
        e = self.alpha_fc(alpha_unpacked)

        def masked_softmax(batch_tensor, mask):
            exp = torch.exp(batch_tensor)
            masked_exp = exp * mask
            sum_masked_exp = torch.sum(masked_exp, dim=1, keepdim=True)
            return masked_exp / sum_masked_exp

        alpha = masked_softmax(e, mask)

        context = torch.bmm(torch.transpose(alpha, 1, 2), alpha_unpacked).squeeze(1)
        logit = self.output(context)
        return logit


 