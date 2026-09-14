"""Inference layers adapted from Music X Lab (MIT; see LICENSE)."""

import torch
from torch import nn
from torch.nn import functional as F

SPEC_DIM = 252


class CNNFeatureExtractor(nn.Module):

    def norm_layer(self, channels):
        return nn.InstanceNorm2d(channels)

    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()

        self.cdim1 = 16
        self.cdim2 = 32
        self.cdim3 = 64
        self.cdim4 = 80

        self.conv1a = nn.Conv2d(1, self.cdim1, (3, 3), padding=(1, 1))
        self.norm1a = self.norm_layer(self.cdim1)
        self.conv1b = nn.Conv2d(self.cdim1, self.cdim1, (3, 3), padding=(1, 1))
        self.norm1b = self.norm_layer(self.cdim1)
        self.conv1c = nn.Conv2d(self.cdim1, self.cdim1, (3, 3), padding=(1, 1))
        self.norm1c = self.norm_layer(self.cdim1)
        self.pool1 = nn.MaxPool2d((1, 3))
        self.conv2a = nn.Conv2d(self.cdim1, self.cdim2, (3, 3), padding=(1, 1))
        self.norm2a = self.norm_layer(self.cdim2)
        self.conv2b = nn.Conv2d(self.cdim2, self.cdim2, (3, 3), padding=(1, 1))
        self.norm2b = self.norm_layer(self.cdim2)
        self.conv2c = nn.Conv2d(self.cdim2, self.cdim2, (3, 3), padding=(1, 1))
        self.norm2c = self.norm_layer(self.cdim2)
        self.pool2 = nn.MaxPool2d((1, 3))
        self.conv3a = nn.Conv2d(self.cdim2, self.cdim3, (3, 3), padding=(1, 1))
        self.norm3a = self.norm_layer(self.cdim3)
        self.conv3b = nn.Conv2d(self.cdim3, self.cdim3, (3, 3), padding=(1, 1))
        self.norm3b = self.norm_layer(self.cdim3)
        self.pool3 = nn.MaxPool2d((1, 4))
        self.conv4a = nn.Conv2d(self.cdim3, self.cdim4, (3, 3), padding=(1, 0))
        self.norm4a = self.norm_layer(self.cdim4)
        self.conv4b = nn.Conv2d(self.cdim4, self.cdim4, (3, 3), padding=(1, 0))
        self.norm4b = self.norm_layer(self.cdim4)
        self.output_size = 3 * self.cdim4

    def forward(self, x):
        assert len(x.shape) == 3
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        x = x.view((batch_size, 1, seq_length, SPEC_DIM))
        x = F.selu(self.norm1a(self.conv1a(x)))
        x = F.selu(self.norm1b(self.conv1b(x)))
        x = F.selu(self.norm1c(self.conv1c(x)))
        x = self.pool1(x)
        x = F.selu(self.norm2a(self.conv2a(x)))
        x = F.selu(self.norm2b(self.conv2b(x)))
        x = F.selu(self.norm2c(self.conv2c(x)))
        x = self.pool2(x)
        x = F.selu(self.norm3a(self.conv3a(x)))
        x = F.selu(self.norm3b(self.conv3b(x)))
        x = self.pool3(x)
        x = F.selu(self.norm4a(self.conv4a(x)))
        x = F.selu(self.norm4b(self.conv4b(x)))
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view((batch_size, seq_length, self.output_size))
        )
        return x


class ChordNet(nn.Module):
    """CNN and bidirectional LSTM with the original six chord-factor heads."""

    def __init__(self):
        super().__init__()
        self.audio_feature_block = CNNFeatureExtractor()
        # Kept for strict compatibility with the published state dictionaries.
        self.condition_linear = nn.Linear(
            self.audio_feature_block.output_size + 12 + 6 + 12, 128
        )
        self.hidden_dim1 = 192
        self.lstm1 = nn.LSTM(
            input_size=self.audio_feature_block.output_size,
            hidden_size=self.hidden_dim1 // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.final_fc1 = nn.Linear(self.hidden_dim1, 100)

    def forward(self, cqt: torch.Tensor):
        features = self.audio_feature_block(cqt)
        batch_size, sequence_length = features.shape[:2]
        hidden = features.new_zeros(2, batch_size, self.hidden_dim1 // 2)
        output, _ = self.lstm1(features, (hidden, hidden.clone()))
        logits = self.final_fc1(output).reshape(batch_size * sequence_length, 100)
        return logits.split((73, 13, 4, 4, 3, 3), dim=-1)

    @torch.inference_mode()
    def inference(self, cqt: torch.Tensor):
        # CQTV2 computes the original pitch-shift margin on either side.
        cropped = cqt[:, 18:270].reshape(1, cqt.shape[0], SPEC_DIM)
        return tuple(
            F.softmax(logits, dim=-1).float().cpu().numpy() for logits in self(cropped)
        )
