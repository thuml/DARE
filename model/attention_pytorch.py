import math

import torch
import torch.nn as nn


class AttentionModel(nn.Module):
    def __init__(self, input_size):
        super(AttentionModel, self).__init__()
        self.main_model = nn.Sequential(
            nn.Linear(input_size, 36),
            nn.PReLU(num_parameters=1, init=0.1),
            nn.Linear(36, 1),
        )

    def forward(self, x):
        return self.main_model(x)


def attention(q, k, v, mask=None, attn_func='learnable', return_scores=False, attn_model=None, time_embedding_dim=0, use_time_mode="concat", item_embedding_dim=None, soft_search=True):
    assert len(q.shape) == 2
    assert len(v.shape) == 3
    assert q.shape[0] == k.shape[0]
    assert q.shape[1] == k.shape[2]
    assert use_time_mode == "concat" or use_time_mode == "add"

    q = q.unsqueeze(1)

    if attn_func == 'learnable':
        assert attn_model is not None
        q = q.repeat(1, k.shape[1], 1)
        concated_input = torch.cat((q, k, q-k, torch.mul(q, k)), dim=2)
        scores = attn_model(concated_input)
        scores = scores / q.shape[2]
        scores = scores.transpose(1, 2)
    elif attn_func == 'scaled_dot_product':
        if use_time_mode == "concat":
            if not soft_search:
                score_for_semantics = torch.matmul(q[:, :, :item_embedding_dim], k[:, :, :item_embedding_dim].transpose(1, 2)) / math.sqrt(float(q[:, :, :item_embedding_dim].shape[2]))
            else:
                score_for_semantics = torch.matmul(q[:, :, :-time_embedding_dim], k[:, :, :-time_embedding_dim].transpose(1, 2)) / math.sqrt(float(q[:, :, :-time_embedding_dim].shape[2]))
            score_for_time = torch.matmul(q[:, :, -time_embedding_dim:], k[:, :, -time_embedding_dim:].transpose(1, 2)) / math.sqrt(float(time_embedding_dim))
            scores = score_for_semantics + score_for_time
        else:  # use_time_mode: add
            scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(float(q.shape[-1]))
    else:
        raise ValueError('Unknown attention function.')

    assert scores.shape == (k.shape[0], 1, k.shape[1])
    scores = scores.squeeze(1)

    if mask is not None:
        assert mask.shape == scores.shape
        mask = mask.type(torch.bool)
        scores = torch.where(mask, scores, torch.ones_like(scores).to(scores.device) * (- 2**32 + 1))

    attention_distribution = nn.functional.softmax(scores, dim=1)

    out = torch.matmul(attention_distribution.unsqueeze(1), v)
    out = out.squeeze(1)

    if return_scores:
        return out, scores, attention_distribution
    else:
        return out
