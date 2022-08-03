import logging

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCELoss
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel

logger = logging.getLogger(__name__)


class QuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        # self.qa_linear = nn.Linear(config.hidden_size, config.num_labels)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                start_logits=None,
                end_logits=None,
                ):
        bert_out = self.bert(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             )[0]
        # logits = self.qa_linear(bert_out)
        logits = self.qa_outputs(bert_out)
        start_predict_logits, end_predict_logits = logits.split(1, dim=-1)
        start_predict_logits = start_predict_logits.squeeze(-1).sigmoid()
        end_predict_logits = end_predict_logits.squeeze(-1).sigmoid()

        total_loss = None
        if start_logits is not None and end_logits is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_logits.size()) > 1:
                start_logits = start_logits.squeeze(-1)
            if len(end_logits.size()) > 1:
                end_logits = end_logits.squeeze(-1)
            start_logits = torch.tensor(start_logits, dtype=torch.float)
            end_logits = torch.tensor(end_logits, dtype=torch.float)

            loss_fct = F.binary_cross_entropy
            start_loss = loss_fct(start_predict_logits, start_logits)
            end_loss = loss_fct(end_predict_logits, end_logits)
            total_loss = (start_loss + end_loss) / 2

        return total_loss, start_predict_logits, end_predict_logits
