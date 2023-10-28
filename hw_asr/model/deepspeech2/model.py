import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.init as init
from typing import Tuple

from hw_asr.model.deepspeech2.convolutions import Convolutions
from hw_asr.model.deepspeech2.rnn import RNNLayer


class DeepSpeech2(nn.Module):
    def __init__(
            self,
            n_feats: int,
            n_class: int,
            num_rnn_layers: int=4,
            rnn_hidden_dim: int=512,
            bidirectional: bool=True,
            **batch
        ):
        super().__init__()
        self.conv = Convolutions(n_feats)
        self.rnn_layers = nn.ModuleList()

        rnn_output_size = rnn_hidden_dim * 2 if bidirectional else rnn_hidden_dim

        for i in range(num_rnn_layers):
            self.rnn_layers.append(
                RNNLayer(
                    input_size=self.conv.output_dim if i == 0 else rnn_output_size,
                    hidden_state_dim=rnn_hidden_dim,
                    bidirectional=bidirectional,
                )
            )

        linear = nn.Linear(rnn_output_size, n_class, bias=False)
        init.xavier_uniform_(linear.weight)
        layer_norm = torch.nn.LayerNorm(rnn_output_size)

        self.fc = nn.Sequential(
            layer_norm,
            linear,
        )

    def forward(
            self,
            spectrogram: Tensor,
            spectrogram_length: Tensor,
            **batch
        ) -> Tuple[Tensor, Tensor]:
        outputs, output_lengths = self.conv(spectrogram, spectrogram_length)

        for rnn_layer in self.rnn_layers:
            outputs = rnn_layer(outputs, output_lengths)

        outputs = self.fc(outputs)
        return outputs
    
    def transform_input_lengths(self, input_lengths):
        lengths = self.conv.get_time_reduce(input_lengths)
        return lengths
    
