import torch
import torch.nn as nn
import torch.nn.functional as F
from polp.nn.layers import AttentionMask

class _Classifier(nn.Module):
    def __init__(self, hidden_size=768, hidden_dropout_prob=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size*2, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, 2)

    def forward(self, features):
        x = features[:, 0, :]
        x = x.reshape(-1, x.size(-1)*2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Model(nn.Module):
    def __init__(self, encoder, block_size):
        super().__init__()
        self.encoder = encoder
        self.classifier = _Classifier()
        self.block_size = block_size
        
    def forward(self, input_ids=None, attention_mask=None ,labels=None):
        input_ids = input_ids.view(-1, self.block_size)
        attention_mask = attention_mask.view(-1, self.block_size)
        attention_mask = AttentionMask(attention_mask)
        outputs = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_layer_state
        logits = self.classifier(outputs)
        prob = F.softmax(logits)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return logits
