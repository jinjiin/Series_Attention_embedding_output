import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from .attention_layers import DownsampledMultiHeadAttention as MultiHeadAttention


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
    def __init__(self, d_model, d_inner, n_head, d_v, dropout=0.1):
        super(selfAttentionLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(out_channels=d_v, embed_dim=d_model, num_heads=n_head, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_v, d_inner, dropout=dropout)

    def forward(self, self_inp, slf_attn_mask=True):
        output, slf_attn = self.slf_attn(query=self_inp, key=self_inp, value=self_inp, mask_future_timesteps=slf_attn_mask)
        output = self.pos_ffn(output)
        return output, slf_attn


class crossAttentionLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_v, dropout=0.1):
        super(crossAttentionLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(out_channels=d_v, embed_dim=d_model, num_heads=n_head, dropout=dropout)
        self.cross_attn = MultiHeadAttention(out_channels=d_v, embed_dim=d_model, num_heads=n_head, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_v, d_inner, dropout=dropout)

    def forward(self, self_inp, cross_inp, slf_attn_mask=True):
        self_output, slf_attn = self.slf_attn(query=self_inp, key=self_inp, value=self_inp, mask_future_timesteps=slf_attn_mask)
        cross_output, cross_attn = self.cross_attn(query=self_output, key=cross_inp, value=cross_inp, mask_future_timesteps=slf_attn_mask)
        output = self.pos_ffn(cross_output)
        return output, slf_attn, cross_attn



class QANet(nn.Module):
    def __init__(self, n_head=4, embedding_dim=50, feature_concat='l_r', d_k=64, attention=True, gate=None, position_embed=False, time_len=24, dis_fcn=False, attn_mask=False):
        super().__init__()
        self.attention = attention
        self.position_embed = position_embed
        self.dis_fcn= dis_fcn
        if self.attention:
            L_d_k = d_k
            R_d_k = d_k
            self.feature_concat = feature_concat
            self.attn_mask = attn_mask
            self.L_self_att1 = selfAttentionLayer(d_model=embedding_dim, d_inner=embedding_dim * 2,
                                        n_head=n_head,  d_v=L_d_k)
            self.R_self_att1 = selfAttentionLayer(d_model=embedding_dim, d_inner=embedding_dim * 2,
                                        n_head=n_head, d_v=R_d_k)
            self.L_self_att2 = selfAttentionLayer(d_model=d_k, d_inner=embedding_dim * 2,
                                                  n_head=n_head, d_v=embedding_dim)
            self.R_self_att2 = selfAttentionLayer(d_model=d_k, d_inner=embedding_dim * 2,
                                                  n_head=n_head, d_v=embedding_dim)

            self.LR_cross_att = crossAttentionLayer(d_model=embedding_dim, d_inner=embedding_dim * 2,
                                          n_head=n_head, d_v=embedding_dim)
            self.RL_cross_att = crossAttentionLayer(d_model=embedding_dim, d_inner=embedding_dim * 2,
                                                    n_head=n_head, d_v=embedding_dim)
            self.LR_cross_att_2 = crossAttentionLayer(d_model=embedding_dim, d_inner=embedding_dim,
                                                    n_head=n_head, d_v=embedding_dim)
            self.RL_cross_att_2 = crossAttentionLayer(d_model=embedding_dim, d_inner=embedding_dim,
                                                    n_head=n_head, d_v=embedding_dim)

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

        self.number_embed = nn.Embedding(1500, embedding_dim)
        # if embedding_dim == 50:  # embed50_no_l2
        #     path = '/lfs1/users/jbyu/NumberEmbedding_0_99/saved/models/TransE/0630_095037/model_best.pth'
        # elif embedding_dim == 128:  # embed128_no_l2
        #     path = '/lfs1/users/jbyu/NumberEmbedding_0_99/saved/models/TransE/0623_112128/model_best.pth'
        # checkpoint = torch.load(path)
        # self.number_embed.load_state_dict({'weight': checkpoint['state_dict']['entityEmbedding.weight']})

        if self.position_embed:
            self.position_embedding_L = nn.Embedding(time_len, embedding_dim)
            self.position_embedding_R = nn.Embedding(time_len, embedding_dim)
            # self.position_embedding_L = SinusoidalPositionalEmbedding(time_len, embedding_dim)
            # self.position_embedding_R = SinusoidalPositionalEmbedding(time_len, embedding_dim)
        if self.dis_fcn:
            self.linear_1 = nn.Linear(in_channel * time_len, in_channel)
            self.linear_2 = nn.Linear(in_channel, 1)

            self.linear_3 = nn.Linear(in_channel * time_len, in_channel)
            self.linear_4 = nn.Linear(in_channel, 1)
        else:
            self.linear_1 = nn.Linear(in_channel * time_len * 2, in_channel)
            self.linear_2 = nn.Linear(in_channel, 2)
        # self.linear_1 = nn.Linear(in_channel * 24 * 2, 2)

    def forward(self, L, R):
        # L = self.number_embed(L)
        # R = self.number_embed(R)

        if self.position_embed:
            batchsize = L.shape[0]
            pos = np.arange(0, 24, 1)
            pos = torch.LongTensor(pos).expand(batchsize, 24).cuda()
            L_pos = self.position_embedding_L(pos)
            R_pos = self.position_embedding_R(pos)
            L = L + L_pos
            R = R + R_pos
        if self.attention:
            L_out, _ = self.L_self_att1(L, self.attn_mask)
            R_out, _ = self.R_self_att1(R, self.attn_mask)

            L_out, _ = self.L_self_att2(L_out, self.attn_mask)
            R_out, _ = self.R_self_att2(R_out, self.attn_mask)

            L_inp, R_inp = None, None
            if self.feature_concat == 'l_r':
                L_inp = L_out
                R_inp = R_out
            else:
                LR_out, _, _ = self.LR_cross_att(L_out, R_out, self.attn_mask)
                LR_out, _, _ = self.LR_cross_att_2(LR_out, R_out, self.attn_mask)

                RL_out, _, _ = self.RL_cross_att(R_out, L_out, self.attn_mask)
                RL_out, _, _ = self.RL_cross_att_2(RL_out, L_out, self.attn_mask)

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
        print('init linear weight-----------------------', classname)
        nn.init.xavier_uniform_(m.weight.data)
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
    c_dim = 50
    q_dim = 50
    context = torch.randn((batchsize, seq_len, c_dim))
    question = torch.randn((batchsize, seq_len, q_dim))

    print(context)
    model = QANet(n_head=1, embedding_dim=50, feature_concat='l_out_direct', d_k=100, attention=True, gate=None)
    L, R = model(context, question)
    print(L.shape, R.shape)
    print(L)

