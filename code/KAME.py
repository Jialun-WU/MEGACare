import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def load_basic_embedding(opt):
    m = np.load(opt.tree_embs)
    w = (m['w'] + m['w_tilde']) / 2.0
    w = np.array(w, dtype='float32')
    w = np.concatenate((w, [np.zeros(opt.dim_emb_basic, dtype='float32')]))
    return w
 

def mask_softmax(opt, batch_tensor, mask, dim=3):
    exp = torch.exp(batch_tensor)

    exp_numpy = exp.cpu().detach().numpy()
    exp_numpy[np.where(exp_numpy == np.inf)] = 1
    exp = Variable(torch.from_numpy(exp_numpy).cuda())
    masked_exp = exp * mask
    sum_masked_exp = torch.sum(masked_exp, dim=dim, keepdim=True)

    if opt.use_cuda:
        sum_masked_exp_numpy = sum_masked_exp.cpu().detach().numpy()
        sum_masked_exp_numpy[np.where(sum_masked_exp_numpy == 0)] = -1
        sum_masked_exp = Variable(torch.from_numpy(sum_masked_exp_numpy).cuda())
    else:
        sum_masked_exp_numpy = sum_masked_exp.detach().numpy()
        sum_masked_exp_numpy[np.where(sum_masked_exp_numpy == 0)] = -1
        sum_masked_exp = Variable(torch.from_numpy(sum_masked_exp_numpy))

    se = (masked_exp / sum_masked_exp).cpu().data.numpy()
    if np.isnan(se).sum() > 0:
        print 'alpha'
    return masked_exp / sum_masked_exp

def knowledge_softmax(hidden, ancestor, mask):
    exp = torch.exp(torch.bmm(hidden, torch.transpose(ancestor, 1, 2)))
    exp_numpy = exp.cpu().detach().numpy()
    exp_numpy[np.where(exp_numpy == np.inf)] = 1
    exp = Variable(torch.from_numpy(exp_numpy).cuda())

    sum_mask_exp = torch.sum(exp, dim=2, keepdim=True)
    masked_exp = exp * mask.unsqueeze(2)
    return masked_exp/sum_mask_exp

class KAME(nn.Module):
    def __init__(self, opt):
        super(KAME, self).__init__()
        self.opt = opt
        self.num_codes = opt.num_codes

        self.basic_embedding = nn.Embedding(self.num_codes + 1, opt.dim_emb_basic)
        self.basic_embedding.weight.data.copy_(torch.from_numpy(load_basic_embedding(opt)))

        self.w_basic = nn.Linear(in_features=opt.dim_emb_basic*2, out_features=opt.dim_emb_basic, bias=False)
        init.xavier_normal(self.w_basic.weight)

        self.u_basic = nn.Linear(in_features=opt.dim_emb_basic, out_features=1)
        init.xavier_normal(self.u_basic.weight)
        self.u_basic.bias.data.zero_()

        self.w_ancestor = nn.Linear(in_features=opt.dim_emb_basic, out_features=opt.dim_emb_basic, bias=True)

        self.RNN = nn.GRU(input_size=opt.dim_emb_basic, hidden_size=opt.dim_hidden_basic,
                                num_layers=opt.num_layer_basic, batch_first=opt.batch_first)

        self.output = nn.Sequential(
            nn.Dropout(p=opt.drop_out),
            nn.Linear(in_features=opt.dim_emb_basic*2, out_features=167, bias=True),
            nn.Sigmoid()
        )

    def forward(self, input):
        seqs, ancestors, length, code_length, ancestor_length = input
        max_visit = torch.max(length).cpu().data
        visit_mask = Variable(torch.FloatTensor([[1.0 if j < i else 0.0 for j in range(max_visit)] for i in length]),
                              requires_grad=False)
        if self.opt.use_cuda:
            visit_mask = visit_mask.cuda()

        seqs_embs = self.basic_embedding(seqs).unsqueeze(3).repeat(1, 1, 1, 6, 1)
        ancestors_embs = self.basic_embedding(ancestors)
        basic_embs = self.u_basic(F.tanh(self.w_basic(torch.cat((seqs_embs, ancestors_embs), 4) * ancestor_length.unsqueeze(4))))
        basic_attention = mask_softmax(self.opt, basic_embs, ancestor_length.unsqueeze(4))
        attention_shape, embs_shape = basic_attention.shape, seqs_embs.shape
        embs_t_v_c = torch.bmm(basic_attention.transpose(4, 3).view(-1, attention_shape[4], attention_shape[3]),
                               ancestors_embs.view(-1, embs_shape[3], embs_shape[4])).\
                               view(embs_shape[0], embs_shape[1], embs_shape[2], -1, embs_shape[4]).squeeze(3)
        embs_t_v = F.tanh(torch.sum(embs_t_v_c, dim=2))
        packed_input_t = pack_padded_sequence(embs_t_v, length, batch_first=self.opt.batch_first)
        g, _ = self.RNN(packed_input_t)
        alpha_unpacked, _ = pad_packed_sequence(g, batch_first=self.opt.batch_first)

        L = self.w_ancestor(ancestors_embs)
        L = torch.sum(torch.sum(L, dim=3), dim=2)[:, :max_visit, :]
        knowledge_attention = knowledge_softmax(alpha_unpacked, L, visit_mask)
        K = torch.bmm(knowledge_attention, L)

        context = torch.cat((K, alpha_unpacked), dim=2)
        context = torch.bmm(visit_mask.unsqueeze(1), context).squeeze(1)
        logit = self.output(context)
        return logit