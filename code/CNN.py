import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, dim_input, dim_emb, kernel_num, kernel_sizes):
        super(CNN, self).__init__()
        Cin = 1
        Cout = kernel_num
        Ks = kernel_sizes

        self.embedding = nn.Embedding(dim_input, dim_emb)
        # self.embedding = nn.Sequential(
        #     nn.Dropout(p=0.1),
        #     nn.Linear(dim_input, 128, bias=False),
        #     nn.Dropout(p=0.1),
        #     nn.ReLU()
        # )

        self.convs = nn.ModuleList([nn.Conv2d(Cin, Cout, (K, dim_emb)) for i, K in enumerate(Ks)])

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(Ks)*Cout, 2)
        self.sigmoid = nn.Sigmoid()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N,Co,W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x).squeeze(3)) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        out = self.fc(x)
        return out
