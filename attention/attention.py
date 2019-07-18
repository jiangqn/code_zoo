import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
import math

INF = 1e9

class Attention(nn.Module):
    """
    The base class of attention.
    """

    def __init__(self, dropout):
        super(Attention, self).__init__()
        self.dropout = dropout

    def forward(self, query, key, value, mask=None):
        """
        :param query: FloatTensor (batch_size, query_size) or FloatTensor (batch_size, num_queries, query_size)
        :param key: FloatTensor (batch_size, time_step, key_size)
        :param value: FloatTensor (batch_size, time_step, value_size)
        :param mask: ByteTensor (batch_size, time_step) or None
        :return output: FloatTensor (batch_size, value_size) or (batch_size, num_queries, value_size)
        """
        single_query = False
        if len(query.size()) == 2:
            query = query.unsqueeze(1)
            single_query = True
        if mask is not None:
            mask = mask.unsqueeze(1)
        score = self._score(query, key) # FloatTensor (batch_size, num_queries, time_step)
        weights = self._weights_normalize(score, mask)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        output = weights.matmul(value)
        if single_query:
            output = output.squeeze(1)
        return output

    def _score(self, query, key):
        """
        :param query: FloatTensor (batch_size, num_queries, query_size)
        :param key: FloatTensor (batch_size, time_step, key_size)
        :return score: FloatTensor (batch_size, num_queries, time_step)
        """
        raise NotImplementedError('Attention score method is not implemented.')

    def _weights_normalize(self, score, mask):
        """
        :param score: FloatTensor (batch_size, num_queries, time_step)
        :param mask: ByteTensor (batch_size, 1, time_step)
        :return weights: FloatTensor (batch_size, num_queries, time_step)
        """
        if not mask is None:
            score = score.masked_fill(mask == 0, -INF)
        weights = F.softmax(score, dim=-1)
        return weights

    def get_attention_weights(self, query, key, mask=None):
        """
        :param query: FloatTensor (batch_size, query_size) or FloatTensor (batch_size, num_queries, query_size)
        :param key: FloatTensor (batch_size, time_step, key_size)
        :param mask: ByteTensor (batch_size, time_step) or None
        :return weights: FloatTensor (batch_size, num_queries, time_step)
        """
        single_query = False
        if len(query.size()) == 2:
            query = query.unsqueeze(1)
            single_query = True
        if mask is not None:
            mask = mask.unsqueeze(1)
        score = self._score(query, key)  # FloatTensor (batch_size, num_queries, time_step)
        weights = self._weights_normalize(score, mask)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        if single_query:
            weights = weights.squeeze(1)
        return weights

class DotAttention(Attention):

    def __init__(self, dropout=0):
        super(DotAttention, self).__init__(dropout)

    def _score(self, query, key):
        assert query.size(2) == key.size(2)
        return query.matmul(key.transpose(1, 2))

class ScaledDotAttention(Attention):

    def __init__(self, dropout=0):
        super(ScaledDotAttention, self).__init__(dropout)

    def _score(self, query, key):
        assert query.size(2) == key.size(2)
        return query.matmul(key.transpose(1, 2)) / math.sqrt(query.size(2))

class BilinearAttention(Attention):

    def __init__(self, query_size, key_size, dropout=0):
        super(BilinearAttention, self).__init__(dropout)
        self.weights = nn.Parameter(torch.FloatTensor(query_size, key_size))
        init.xavier_uniform_(self.weights)

    def _score(self, query, key):
        score = query.matmul(self.weights).matmul(key.transpose(1, 2))
        return score

class ConcatAttention(Attention):

    def __init__(self, query_size, key_size, dropout=0):
        super(ConcatAttention, self).__init__(dropout)
        self.query_weights = nn.Parameter(torch.Tensor(query_size, 1))
        self.key_weights = nn.Parameter(torch.Tensor(key_size, 1))
        init.xavier_uniform_(self.query_weights)
        init.xavier_uniform_(self.key_weights)

    def _score(self, query, key):
        batch_size, num_queries, time_step = query.size(0), query.size(1), key.size(1)
        query = query.matmul(self.query_weights).expand(batch_size, num_queries, time_step)
        key = key.matmul(self.key_weights).transpose(1, 2).expand(batch_size, num_queries, time_step)
        score = query + key
        return score

class MlpAttention(Attention):

    def __init__(self, query_size, key_size, out_size=10, dropout=0):
        super(MlpAttention, self).__init__(dropout)
        self.query_projection = nn.Linear(query_size, out_size)
        self.key_projection = nn.Linear(key_size, out_size)
        self.v = nn.Parameter(torch.FloatTensor(out_size, 1))
        init.xavier_uniform_(self.v)

    def _score(self, query, key):
        batch_size, num_queries, time_step, out_size = query.size(0), query.size(1), key.size(1), self.v.size(0)
        query = self.query_projection(query).unsqueeze(-1).expand(batch_size, num_queries, time_step, out_size)
        key = self.key_projection(key).unsqueeze(1).expand(batch_size, num_queries, time_step, out_size)
        score = torch.tanh(query + key).matmul(self.v).squeeze(-1)
        return score

class TanhBilinearAttention(Attention):

    def __init__(self, query_size, key_size, dropout=0):
        super(TanhBilinearAttention, self).__init__(dropout)
        self.weights = nn.Parameter(torch.FloatTensor(query_size, key_size))
        init.xavier_uniform_(self.weights)
        self.bias = nn.Parameter(torch.zeros(1))

    def _score(self, query, key):
        score = torch.tanh(query.matmul(self.weights).matmul(key.transpose(1, 2)) + self.bias)
        return score

class TanhConcatAttention(Attention):

    def __init__(self, query_size, key_size, dropout=0):
        super(TanhConcatAttention, self).__init__(dropout)
        self.query_weights = nn.Parameter(torch.Tensor(query_size, 1))
        self.key_weights = nn.Parameter(torch.Tensor(key_size, 1))
        init.xavier_uniform_(self.query_weights)
        init.xavier_uniform_(self.key_weights)

    def _score(self, query, key):
        batch_size, num_queries, time_step = query.size(0), query.size(1), key.size(1)
        query = query.matmul(self.query_weights).expand(batch_size, num_queries, time_step)
        key = key.matmul(self.key_weights).transpose(1, 2).expand(batch_size, num_queries, time_step)
        score = query + key
        score = torch.tanh(score)
        return score

def get_attention(query_size, key_size, attention_type='Bilinear'):
    if attention_type == 'Dot':
        attention = DotAttention()
    elif attention_type == 'ScaledDot':
        attention = ScaledDotAttention()
    elif attention_type == 'Concat':
        attention = ConcatAttention(
            query_size=query_size,
            key_size=key_size
        )
    elif attention_type == 'Bilinear':
        attention = BilinearAttention(
            query_size=query_size,
            key_size=key_size
        )
    elif attention_type == 'MLP':
        attention = MlpAttention(
            query_size=query_size,
            key_size=query_size,
            out_size=10
        )
    elif attention_type == 'TanhConcat':
        attention = ConcatAttention(
            query_size=query_size,
            key_size=key_size
        )
    elif attention_type == 'TanhBilinear':
        attention = BilinearAttention(
            query_size=query_size,
            key_size=key_size
        )
    else:
        raise ValueError('%s attention is not supported.' % attention_type)
    return attention