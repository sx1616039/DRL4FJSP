import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, d_model, num_output, num_head=2, dim_ff=256, num_layers=1, dropout=0.1):
        super(Actor, self).__init__()
        assert d_model % num_head == 0, "the number of heads must divide evenly into d_model"
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_head, dim_feedforward=dim_ff,
                                                   dropout=dropout, batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.action_head = nn.Linear(d_model, num_output)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self, d_model, num_output=1, num_head=2, dim_ff=256, num_layers=1, dropout=0.1):
        super(Critic, self).__init__()
        assert d_model % num_head == 0, "the number of heads must divide evenly into d_model"
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_head, dim_feedforward=dim_ff,
                                                   dropout=dropout, batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.state_value = nn.Linear(d_model, num_output)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        value = self.state_value(x)
        return value

