import logging

import torch
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss, BCELoss
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel

logger = logging.getLogger(__name__)


class Attention(nn.Module):
    """Attention from 'Attention Is All You Need' paper"""

    def __init__(self, embed_dim, bias=True, ):
        super().__init__()
        self.k_proj = nn.Linear(embed_dim, 32, bias=bias)
        self.q_proj = nn.Linear(embed_dim, 32, bias=bias)

    def forward(self,
                query: Tensor,
                key: Tensor,
                ) -> Tensor:
        """Input shape: Batch x Time(SeqLen) x Channel"""
        bsz, tgt_len, embed_dim = query.size()
        # todo only use q*W*k
        query = self.q_proj(query)  # B, Q, H
        key = self.k_proj(key)  # B, K, H
        src_len = key.size(1)

        attn_weights = torch.bmm(query, key.transpose(1, 2))  # B, Q, K
        assert attn_weights.size() == (bsz, tgt_len, src_len)

        # attn_weights = F.softmax(attn_weights, dim=-1)

        return attn_weights


class Pointer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.us = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.vh = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.out = nn.Linear(config.hidden_size, 2, bias=True)

    def forward(self, hp, sp):
        h = torch.mul(sp, hp) + self.us(sp) + self.vh(hp)
        return self.out(h)


class QuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.AttentionES = Attention(config.hidden_size)
        self.AttentionSE = Attention(config.hidden_size)
        # self.Pointer = Pointer(config)

        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                starts=None,
                ends=None,
                masks=None,
                ):
        bert_out = self.bert(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             )[0]
        # embeddings = self.bert.embeddings(input_ids)
        embeddings = bert_out

        if starts is not None:
            # If we are on multi-GPU, split add a dimension
            if len(starts.size()) > 1:
                starts = starts.squeeze(-1)
            if len(ends.size()) > 1:
                ends = ends.squeeze(-1)
            if len(masks.size()) > 1:
                masks = masks.squeeze(-1)

            batchSize = ends.size(0)
            loss_fct = CrossEntropyLoss()
            # E-S
            end = ends[:, :-1] * masks
            Q = torch.stack([embeddings[i].index_select(dim=0, index=end[i]) for i in range(batchSize)], dim=0)
            attentionES = self.AttentionES(Q, bert_out)
            lossES = loss_fct(attentionES.transpose(1, 2), starts)
            # S-E
            start = starts * masks
            Q = torch.stack([embeddings[i].index_select(dim=0, index=start[i]) for i in range(batchSize)], dim=0)
            attentionSE = self.AttentionSE(Q, bert_out)
            lossSE = loss_fct(attentionSE.transpose(1, 2), ends[:, 1:])
            loss = lossES + lossSE
            return loss, None

        attentionES = self.AttentionES(embeddings, bert_out)
        attentionSE = self.AttentionSE(embeddings, bert_out)

        # todo 改这个地方，保证结果有效
        ES_logits = torch.argmax(attentionES, dim=2)
        SE_logits = torch.argmax(attentionSE, dim=2)
        return None, ES_logits, SE_logits
