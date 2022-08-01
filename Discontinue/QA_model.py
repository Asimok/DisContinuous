import logging

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCELoss
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel

logger = logging.getLogger(__name__)


# class Pointer(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#
#         self.swh = nn.Linear(512, 2)
#         self.u1s = nn.Linear(config.hidden_size, 2, bias=False)
#         self.u2h = nn.Linear(config.hidden_size, 2, bias=False)
#
#     def forward(self, hp, sp):
#         score = self.swh(torch.matmul(sp, hp.transpose(1, 2))) + self.u1s(hp) + self.u2h(sp)
#         return score


class Pointer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.sw = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        # self.hw = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        # self.us = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        # self.vh = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        # self.out = nn.Linear(config.hidden_size, 2, bias=True)

        # self.shw = nn.Linear(config.hidden_size, 2, bias=True)
        # self.us = nn.Linear(config.hidden_size, 2, bias=False)
        # self.vh = nn.Linear(config.hidden_size, 2, bias=False)

        self.us = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.vh = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.out = nn.Linear(config.hidden_size, 2, bias=True)

    def forward(self, hp, sp):
        # h = torch.mul(self.sw(sp), self.hw(hp)) + self.us(sp) + self.vh(hp)
        # return self.out(h)

        # return self.shw(torch.mul(sp, hp)) + self.us(sp) + self.vh(hp)

        h = torch.mul(sp, hp) + self.us(sp) + self.vh(hp)
        return self.out(h)


class QuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        # TODO
        #  why use LSTM? I hope to delete it.
        self.decoder_lstm = nn.LSTM(input_size=config.hidden_size * 2,
                                    hidden_size=config.hidden_size,
                                    num_layers=2,
                                    batch_first=True,
                                    bias=True)
        self.Pointer = Pointer(config)
        self.qa_checks = nn.Linear(config.hidden_size, 3)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                BIOs=None,
                checks=None,
                ):
        bert_out = self.bert(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             )[0]
        embeddings = self.bert.embeddings(input_ids)

        decoder_input = torch.cat((embeddings, bert_out), dim=2)
        decoder_output = self.decoder_lstm(decoder_input)[0]

        P_logits = self.Pointer(hp=bert_out, sp=decoder_output)
        C_logits = self.qa_checks(bert_out)

        loss = None
        if BIOs is not None:
            # If we are on multi-GPU, split add a dimension
            if len(BIOs.size()) > 1:
                BIOs = BIOs.squeeze(-1)
            if len(checks.size()) > 1:
                checks = checks.squeeze(-1)

            loss_fct = CrossEntropyLoss()
            loss1 = loss_fct(P_logits.transpose(1, 2), BIOs)
            loss2 = loss_fct(C_logits.transpose(1, 2), checks)
            # loss = loss1 + loss2
            loss = loss1
            return loss, P_logits
        P_logits = torch.argmax(P_logits, dim=2)
        logits = torch.argmax(C_logits, dim=2)
        return loss, P_logits, logits
