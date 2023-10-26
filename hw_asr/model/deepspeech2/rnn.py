import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RNNLayer(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_state_dim: int = 512,
            bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_state_dim = hidden_state_dim
        self.relu = F.relu
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_state_dim,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0.1,
            bidirectional=bidirectional,
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor):
        inputs = self.relu(self.batch_norm(inputs.transpose(1, 2)))
        inputs = inputs.transpose(1, 2)

        # print('GRU input shape:', inputs.shape)
        packed = nn.utils.rnn.pack_padded_sequence(inputs, input_lengths.cpu(), enforce_sorted=False, batch_first=True)
        outputs, hidden_states = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        # print('GRU output shape', outputs.shape)
        return outputs