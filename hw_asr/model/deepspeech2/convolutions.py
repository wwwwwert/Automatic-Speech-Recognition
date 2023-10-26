import torch.nn as nn
from torch import Tensor
from typing import Tuple


class Convolutions(nn.Module):
    def __init__(
            self,
            input_dim: int,
            in_channels: int = 1,
            out_channels: int = 32,
        ):
        super().__init__()
        self.input_dim = input_dim
        self.activation = nn.ReLU()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5), bias=False),
            nn.BatchNorm2d(out_channels),
            self.activation,
            nn.Conv2d(out_channels, out_channels, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5), bias=False),
            nn.BatchNorm2d(out_channels),
            self.activation,
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        output, output_lengths = self.apply_convs(inputs.unsqueeze(1), input_lengths)

        batch_size, channels, dimension, seq_lengths = output.shape
        output = output.permute(0, 3, 1, 2)
        output = output.view(batch_size, seq_lengths, channels * dimension)

        return output, output_lengths
    
    def apply_convs(self, inputs: Tensor, seq_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        for module in self.convs:
            output = module(inputs)
            seq_lengths = self.calc_lengths(module, seq_lengths)
            inputs = output

        return output, seq_lengths
    
    def calc_lengths(self, module: nn.Module, seq_lengths: Tensor) -> Tensor:
        if isinstance(module, nn.Conv2d):
            numerator = seq_lengths + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1
            seq_lengths = numerator.float() / float(module.stride[1])
            seq_lengths = seq_lengths.int() + 1
            
        elif isinstance(module, nn.MaxPool2d):
            seq_lengths //= 2

        return seq_lengths.int()

    @property
    def output_dim(self):
        output_dim = self.input_dim
        for module in self.convs:
            output_dim = self.calc_dim(module, output_dim)
        output_dim *= self.out_channels
        return output_dim
    
    def calc_dim(self, module: nn.Module, input_dim: int) -> int:
        if isinstance(module, nn.Conv2d):
            numerator = input_dim + 2 * module.padding[0] - module.dilation[0] * (module.kernel_size[0] - 1) - 1
            input_dim = float(numerator) / float(module.stride[0])
            input_dim = int(input_dim) + 1

        elif isinstance(module, nn.MaxPool2d):
            input_dim //= 2

        return int(input_dim)

    def get_time_reduce(self, input_lengths):
        final_length = input_lengths
        for module in self.convs:
            final_length = self.calc_lengths(module, final_length)
        return final_length