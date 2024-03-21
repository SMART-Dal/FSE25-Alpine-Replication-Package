import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(768, 1)
        
    def forward(self, input_ids=None, attention_mask=None ,labels=None):
        # Perform mean pooling
        #x = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_layer_state.sum(axis=1)
        # Take the <s> embedding
        x = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_layer_state[:, 0, :]
        #n = attention_mask.bool_mask.clone().squeeze(1).squeeze(1).sum(axis=-1).unsqueeze(-1)
        #x = x / n
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = self.out_proj(x)
        prob = F.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10)*labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            return loss, logits
        else:
            return prob