import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(self, input_size, attention_size):
        super().__init__()
        self.attn = nn.Linear(input_size, attention_size)
        self.v = nn.Linear(attention_size, 1, bias=False)

    def forward(self, src, src_mask):
        # src = (batch, seq_len, input_size) input_size = 1
        # src_mask = (batch, seq_len)

        # energy = (batch, seq_len, attention_size)
        energy = torch.tanh(self.attn(src))

        # attention = (batch, seq_len, 1)
        # attention = (batch, seq_len)
        energy = self.v(energy).squeeze()

        energy = energy.masked_fill(src_mask == 0, -1e10)

        # (batch, seq_len)
        attention_weights = F.softmax(energy, dim=1)

        # (batch, seq_len, 1)
        attention_weights = attention_weights.unsqueeze(2)

        context = attention_weights * src
        # context = (batch, seq_len, input_size)

        # return weights
        return context, attention_weights


class PositionwiseFeedforwardLayer(nn.Module):

    def __init__(self, input_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(input_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, seq len, input_dim]

        x = self.dropout(torch.relu(self.fc_1(x)))
        # x = [batch size, seq len, pf dim]

        x = self.fc_2(x)
        # x = [batch size, seq len, input_dim]

        return x


class Aggregation_Attention(nn.Module):

    def __init__(self, pre_size, attention_size):
        super().__init__()
        self.attn = nn.Linear(pre_size, attention_size)
        self.v = nn.Linear(attention_size, 1, bias=False)

    def forward(self, src, src_mask):
        # src = (batch, seq_len, pre_size)
        # src_mask = (batch, seq_len)

        # energy = (batch, seq_len, attention_size)
        energy = torch.tanh(self.attn(src))

        # energy = (batch, seq_len, 1)
        energy = self.v(energy).squeeze()
        # energy = (batch, seq_len)

        energy = energy.masked_fill(src_mask == 0, -1e10)

        # (batch, seq_len)
        attention_weights = F.softmax(energy, dim=1)
        # attention_weight.append(attention_weights)

        # (batch, 1, seq_len)
        attention_weights = attention_weights.unsqueeze(1)

        context = torch.bmm(attention_weights, src).squeeze()
        # context = (batch, pre_size)

        # return weights
        return context, attention_weights


if __name__ == "__main__":
    a = torch.randn(10, 3321, 12)
    a_mask = torch.zeros(10, 3321)
    attn_layer = Attention(12, 24)
    context, weights = attn_layer(a, a_mask)
    print(context.shape)  # torch.Size([10, 3321, 12])
    print(weights.shape)  # torch.Size([10, 3321, 1])

    pf_layer = PositionwiseFeedforwardLayer(12, 32, 0.1)
    a = pf_layer(context)  # torch.Size([10, 3321, 12])
    print(a.shape)
