import logging

import torch
from torch import nn, Tensor, binary_cross_entropy_with_logits
from torch.nn import CrossEntropyLoss, BCELoss
from torch.nn.functional import binary_cross_entropy

from multi_head_attention import MultiHeadAttention
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


# 多头attention

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
        # 多头attention
        self.Attention = MultiHeadAttention(768, 768, 512, 2)
        # self.Attention = nn.MultiheadAttention(embed_dim=768, num_heads=2, kdim=768, vdim=768)

        # self.AttentionES = Attention(config.hidden_size)
        # self.AttentionSE = Attention(config.hidden_size)
        # self.Pointer = Pointer(config)
        # self.qa_outputs = nn.Linear(config.hidden_size, 2)
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
        embeddings = bert_out
        # att, score = self.Attention(embeddings, bert_out)
        # logits = self.qa_outputs(att).softmax(dim=2)
        # start_predict_logits, end_predict_logits = logits.split(1, dim=-1)
        # start_predict_logits = start_predict_logits.squeeze(-1).sigmoid()
        # end_predict_logits = end_predict_logits.squeeze(-1).sigmoid()

        total_loss = None

        if starts is not None:
            # If we are on multi-GPU, split add a dimension
            if len(starts.size()) > 1:
                starts = starts.squeeze(-1)
            if len(ends.size()) > 1:
                ends = ends.squeeze(-1)
            if len(masks.size()) > 1:
                masks = masks.squeeze(-1)
            batchSize = ends.size(0)
            start = starts * masks
            end = ends[:, :-1] * masks
            # Q_s = torch.stack([start_predict_logits[i].index_select(dim=0, index=start[i]) for i in range(batchSize)],
            #                   dim=0).squeeze()
            #
            # Q_e = torch.stack([end_predict_logits[i].index_select(dim=0, index=end[i]) for i in range(batchSize)],
            #                   dim=0).squeeze()
            # loss_fct = binary_cross_entropy
            # start_loss = loss_fct(Q_s.float(), torch.where(starts > 0, 1, 0).float())
            # end_loss = loss_fct(Q_e.float(), torch.where(ends[:, 1:] > 0, 1, 0).float())
            # total_loss = (start_loss + end_loss) / 2

            loss_fct = CrossEntropyLoss()
            # E-S
            Q_s = torch.stack([embeddings[i].index_select(dim=0, index=end[i]) for i in range(batchSize)], dim=0)
            Q_e = torch.stack([embeddings[i].index_select(dim=0, index=start[i]) for i in range(batchSize)], dim=0)

            att, score = self.Attention(Q_s+Q_e, bert_out)
            lossES = loss_fct(score[0].transpose(1, 2), starts)
            lossSE = loss_fct(score[1].transpose(1, 2), ends[:, 1:])
            loss = lossES + lossSE
            return lossES, None
        att, score = self.Attention(embeddings, bert_out)
        ES_logits = torch.argmax(score[0], dim=2)
        SE_logits = torch.argmax(score[1], dim=2)
        return None, ES_logits, SE_logits
