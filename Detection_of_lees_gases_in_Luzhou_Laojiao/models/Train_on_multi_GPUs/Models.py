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


class EncoderLayer(nn.Module):

    def __init__(self,
                 input_size,
                 pre_size,
                 attention_size,
                 pf_dim,
                 dropout):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(pre_size)
        self.ff_layer_norm = nn.LayerNorm(pre_size)
        self.self_attention = Attention(pre_size, attention_size)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(pre_size,
                                                                     pf_dim,
                                                                     dropout)
        self.pre_process = nn.Linear(input_size, pre_size)
        self.dropout = nn.Dropout(dropout)
        # self.attention_weights = attention_weights

    def forward(self, src, src_mask, attention_weights):
        # src = [batch size, src_len, input_dim]
        # src_mask = [batch size, src len]

        # src = [batch, seq len, pre_size]
        src = self.pre_process(src)

        # self attention
        _src, weights = self.self_attention(src, src_mask)
        attention_weights.append(weights)
        # src = [batch, seq_len, input_dim]

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        # positionwise feedforward
        _src = self.positionwise_feedforward(src)

        # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        return src, attention_weights


class Encoder(nn.Module):

    # 目前先用一层的encoder试试

    def __init__(self,
                 n_layers,
                 input_size,
                 pre_size,
                 attention_size,
                 pf_dim,
                 dropout):
        super().__init__()

        # self.device = device_list

        self.layers = nn.ModuleList([EncoderLayer(input_size,
                                                  pre_size,
                                                  attention_size,
                                                  pf_dim,
                                                  dropout)
                                     for i in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mask, attention_weights):
        for number, layer in enumerate(self.layers):
            x, attention_weights = layer(x, x_mask, attention_weights)

        return x, attention_weights


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


class DecoderLayer(nn.Module):

    def __init__(self,
                 pre_size,
                 attention_size
                 ):
        super().__init__()

        self.ag_attn_layer = Aggregation_Attention(pre_size, attention_size)

        self.fc1 = nn.Linear(in_features=pre_size, out_features=196)
        nn.init.xavier_normal_(self.fc1.weight)

        self.fc2 = nn.Linear(in_features=196, out_features=774)
        nn.init.xavier_normal_(self.fc2.weight)

        self.fc3 = nn.Linear(in_features=774, out_features=211)
        nn.init.xavier_normal_(self.fc3.weight)

        self.fc4 = nn.Linear(in_features=211, out_features=6)
        nn.init.xavier_normal_(self.fc4.weight)

    def forward(self, x, x_mask):
        # x = [batch, seq_len, pre_size]
        # x_mask = [batch, seq_len]
        x, attention_weight = self.ag_attn_layer(x, x_mask)

        x = self.fc1(x)
        # x = self.bn1(x)
        x = torch.sigmoid(x)

        x = self.fc2(x)
        # x = self.bn2(x)
        x = torch.sigmoid(x)
        # x = self.dropout1(x)

        x = self.fc3(x)
        x = torch.sigmoid(x)

        x = self.fc4(x)
        x = torch.sigmoid(x)

        return x, attention_weight


class Decoder(nn.Module):

    def __init__(self,
                 pre_size,
                 attention_size
                 ):
        super().__init__()

        # self.device = device_list

        self.layers = DecoderLayer(pre_size, attention_size)

    def forward(self, x, x_mask):

        x, attention_weights = self.layers(x, x_mask)

        return x, attention_weights


class Spec_transformer(nn.Module):

    def __init__(self, Encoder, Decoder):
        super().__init__()

        self.encoder = Encoder
        self.decoder = Decoder

    def forward(self, x, x_mask, weight_list):
        x, weight_list = self.encoder(x, x_mask, weight_list)
        out, weights_2 = self.decoder(x, x_mask)
        weight_list.append(weights_2)

        return out, weight_list


if __name__ == "__main__":

    # 检查GPU可用性
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU可用")
    else:
        device = torch.device("cpu")
        print("没有GPU，将使用CPU")

    a = torch.randn(10, 3321, 1)
    a_mask = torch.zeros(10, 3321)
    attention_weights = []

    # enc_layer = EncoderLayer(1, 12, 24, 32, 0.1)
    # context, weights = enc_layer(a, a_mask, attention_weights)
    # print(context.shape)  # torch.Size([10, 3321, 12])
    # print(weights[0].shape)  # torch.Size([10, 3321, 1])

    encoder = Encoder(1, 1, 12, 24, 32, 0.1)
    # encoded_a, weights = encoder(a, a_mask, attention_weights)
    # print(encoded_a.shape)  # torch.Size([10, 3321, 12])
    # print(len(weights))
    # print(weights[0].shape)  # torch.Size([10, 3321, 1])
    #
    # # decoder_layer = DecoderLayer(12, 32)
    # # decoded_a, weight = decoder_layer(encoded_a, a_mask)
    # # weights.append(weight)
    # # print(decoded_a.shape)
    # # print(len(weights))
    # # print(weights[1].shape)
    #
    decoder = Decoder(12, 32)
    # decoded_a, weight = decoder(encoded_a, a_mask)
    # weights.append(weight)
    # print(decoded_a.shape)
    # print(len(weights))
    # print(weights[1].shape)

    spectrans = Spec_transformer(encoder, decoder)

    spectrans = spectrans.to(device)

    pred, weights = spectrans(a.to(device), a_mask.to(device), attention_weights)
    print(pred.shape)
    print(len(weights))

