import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torch
import math
import numpy as np


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        print(self.pos_table.shape)
        pos = self.pos_table[:, :x.size(1)].clone().detach()
        print(pos.shape, x.shape)
        return x + pos


class SinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions, embedding_dim, padding_idx=None):
        super().__init__(num_positions, embedding_dim)
        if embedding_dim % 2 != 0:
            raise NotImplementedError(f"odd embedding_dim {embedding_dim} not supported")
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """Identical to the XLM create_sinusoidal_embeddings except features are not interleaved.
            The cos features are in the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out[:, 0 : dim // 2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))  # This line breaks for odd n_pos
        out[:, dim // 2 :] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        out.requires_grad = False
        return out

    @torch.no_grad()
    def forward(self, input_ids, use_cache=False):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input_ids.shape[:2]
        if use_cache:
            positions = input_ids.data.new(1, 1).fill_(seq_len - 1)  # called before slicing
        else:
            # starts at 0, ends at 1-seq_len
            positions = torch.arange(seq_len, dtype=torch.long, device=self.weight.device)
        return super().forward(positions)

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        q = self.w_qs(q)
        k = self.w_ks(k)
        v = self.w_vs(v)

        q = q.view(sz_b, len_q, n_head, d_k)
        k = k.view(sz_b, len_k, n_head, d_k)
        v = v.view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))

        out = residual + q
        out = self.layer_norm(out)
        return out, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)

        x += residual
        x = self.layer_norm(x)

        return x


class selfAttentionLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(selfAttentionLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, self_inp, slf_attn_mask=None):
        output, slf_attn = self.slf_attn(q=self_inp, k=self_inp, v=self_inp, mask=slf_attn_mask)
        output = self.pos_ffn(output)
        return output, slf_attn


class crossAttentionLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(crossAttentionLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.cross_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, self_inp, cross_inp, slf_attn_mask=None):
        self_output, slf_attn = self.slf_attn(q=self_inp, k=self_inp, v=self_inp, mask=slf_attn_mask)
        cross_output, cross_attn = self.cross_attn(q=self_output, k=cross_inp, v=cross_inp, mask=slf_attn_mask)
        output = self.pos_ffn(cross_output)
        return output, slf_attn, cross_attn


class QANet(nn.Module):
    def __init__(self, n_head=4, embedding_dim=50, feature_concat='l_r', d_k=64, attention=True, gate=None, position_embed=False, time_len=24, dis_fcn=False):
        super().__init__()
        self.attention = attention
        if self.attention:
            L_d_k = d_k
            R_d_k = d_k
            self.feature_concat = feature_concat
            self.L_self_att1 = selfAttentionLayer(d_model=embedding_dim, d_inner=embedding_dim*2,
                                        n_head=n_head, d_k=L_d_k, d_v=L_d_k)
            self.R_self_att1 = selfAttentionLayer(d_model=embedding_dim, d_inner=embedding_dim*2,
                                        n_head=n_head, d_k=R_d_k, d_v=R_d_k)
            self.L_self_att2 = selfAttentionLayer(d_model=embedding_dim, d_inner=embedding_dim * 2,
                                                  n_head=n_head, d_k=L_d_k, d_v=L_d_k)
            self.R_self_att2 = selfAttentionLayer(d_model=embedding_dim, d_inner=embedding_dim * 2,
                                                  n_head=n_head, d_k=R_d_k, d_v=R_d_k)

            self.LR_cross_att = crossAttentionLayer(d_model=embedding_dim, d_inner=embedding_dim * 2,
                                          n_head=n_head, d_k=L_d_k, d_v=L_d_k)
            self.LR_cross_att_2 = crossAttentionLayer(d_model=embedding_dim, d_inner=embedding_dim * 2,
                                                    n_head=n_head, d_k=L_d_k, d_v=L_d_k)
            self.RL_cross_att = crossAttentionLayer(d_model=embedding_dim, d_inner=embedding_dim * 2,
                                          n_head=n_head, d_k=R_d_k, d_v=R_d_k)
            self.RL_cross_att_2 = crossAttentionLayer(d_model=embedding_dim, d_inner=embedding_dim * 2,
                                                    n_head=n_head, d_k=R_d_k, d_v=R_d_k)

            in_channel = 0
            if self.feature_concat == 'l_r':  # L：L;R:R
                in_channel = embedding_dim
            elif self.feature_concat == 'l_lr':  # L:L_lr;R:R_rl
                in_channel = embedding_dim * 2
            elif self.feature_concat == 'l_out_lr':
                in_channel = embedding_dim * 2
                self.gate_choise = gate
                if self.gate_choise:  # gate: False、self、cross
                    in_channel = embedding_dim
                    self.gate_L = nn.Linear(embedding_dim * 2, embedding_dim)
                    self.gate_R = nn.Linear(embedding_dim * 2, embedding_dim)
            elif self.feature_concat == 'l_out_direct':
                in_channel = embedding_dim
            elif self.feature_concat == 'l_lr_rl':  # L:L_lr_rl; R:r_lr_rl
                in_channel = embedding_dim * 3
        else:
            in_channel = embedding_dim

        if self.position_embed:
            # self.position_embedding_L = nn.Embedding(time_len, embedding_dim)
            # self.position_embedding_R = nn.Embedding(time_len, embedding_dim)
            self.pos_embedding = SinusoidalPositionalEmbedding(time_len, embedding_dim)
        if self.dis_fcn:
            self.linear_1 = nn.Linear(in_channel * time_len, in_channel)
            self.linear_2 = nn.Linear(in_channel, 1)

            self.linear_3 = nn.Linear(in_channel * time_len, in_channel)
            self.linear_4 = nn.Linear(in_channel, 1)
        else:
            self.linear_1 = nn.Linear(in_channel * time_len * 2, in_channel)
            self.linear_2 = nn.Linear(in_channel, 2)


    def forward(self, L, R):
        if self.pos_embedding:
            L_pos = self.pos_embedding(L)
            R_pos = self.pos_embedding(R)
            L = L + L_pos
            R = R + R_pos
        if self.attention:
            L_out, _ = self.L_self_att1(L)
            R_out, _ = self.R_self_att1(R)
            L_out, _ = self.L_self_att2(L_out)
            R_out, _ = self.R_self_att2(R_out)

            L_inp, R_inp = None, None
            if self.feature_concat == 'l_r':
                L_inp = L_out
                R_inp = R_out
            else:
                LR_out, _, _ = self.LR_cross_att(L_out, R_out)
                LR_out, _, _ = self.LR_cross_att_2(LR_out, R_out)

                RL_out, _, _ = self.RL_cross_att(R_out, L_out)
                RL_out, _, _ = self.RL_cross_att_2(RL_out, L_out)

                if self.feature_concat == 'l_lr':
                    L_inp = torch.cat((L, LR_out), dim=2)
                    R_inp = torch.cat((R, RL_out), dim=2)
                elif self.feature_concat == 'l_out_lr':
                    L_inp = torch.cat((L_out, LR_out), dim=2)
                    R_inp = torch.cat((R_out, RL_out), dim=2)

                    if self.gate_choise:
                        g_L = F.sigmoid(self.gate_L(L_inp))
                        g_R = F.sigmoid(self.gate_R(R_inp))

                        # print(g_L.shape, g_R.shape, L_out.shape, R_out.shape)
                        if self.gate_choise == 'self':
                            L_inp = LR_out + g_L * L_out
                            R_inp = RL_out + g_R * R_out
                        elif self.gate_choise == 'cross':
                            L_inp = L_out + g_L * LR_out
                            R_inp = R_out + g_R * RL_out
                    else:
                        L_inp = torch.cat((L_out, LR_out), dim=2)
                        R_inp = torch.cat((R_out, RL_out), dim=2)
                elif self.feature_concat == 'l_lr_rl':
                    L_inp = torch.cat((L, LR_out, RL_out), dim=2)
                    R_inp = torch.cat((R, LR_out, RL_out), dim=2)
                elif self.feature_concat == 'l_out_direct':
                    L_inp, R_inp = LR_out, RL_out
        else:
            L_inp, R_inp = L, R

        if self.dis_fcn:
            L_inp = L_inp.view((L_inp.shape[0], -1))
            R_inp = R_inp.view((R_inp.shape[0], -1))
            L_out = self.linear_2(F.relu(self.linear_1(L_inp)))
            R_out = self.linear_4(F.relu(self.linear_3(R_inp)))
        else:
            inp = torch.cat((L_inp, R_inp), dim=2)
            inp = inp.view((inp.shape[0], -1))
            out = self.linear_2(F.relu(self.linear_1(inp)))
            L_out, R_out = torch.chunk(out, 2, dim=1)
        return L_out, R_out


# custom weights initialization
def weights_init(m):  # define the initialization function
    torch.cuda.manual_seed_all(0)
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif hasattr(m, 'weight') and classname.find('Linear') != -1:
        print('init linear weight-----------------------')
        nn.init.xavier_uniform_(m.weight.data)
        # nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    batchsize = 128
    seq_len = 24
    c_dim = 1
    q_dim = 1
    # context = torch.randn((batchsize, seq_len, c_dim))
    # question = torch.randn((batchsize, seq_len, q_dim))
    pos = np.arange(0, 24, 1)
    L_pos = torch.LongTensor(pos).expand(batchsize, 24)
    print(pos)
    print(L_pos)
    # print(context.shape)

    """ test multi head attention"""
    # model = MultiHeadAttention(n_head=5, d_model=50, d_k=10, d_v=10)
    # q, attention = model(context, context, context)
    # print(q.shape, attention.shape)

    # model = QANet(n_head=4, embedding_dim=128, feature_concat='l_out_direct', d_k=64, attention=True, gate=None)
    # out = model(context, question)
    # print(out.shape)

    # pos_model = PositionalEncoding(d_hid=128, n_position=24)
    pos_model = SinusoidalPositionalEmbedding(num_positions=24, embedding_dim=128)
    y = pos_model(L_pos)
    print(y.shape)

