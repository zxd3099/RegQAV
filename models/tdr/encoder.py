# encoding: utf-8
import torch
import torch.nn as nn

from einops.layers.torch import Rearrange


class GLU(nn.Module):
    def __init__(self, in_channels, out_channels, activation: str = "sigmoid"):
        super(GLU, self).__init__()

        self.linear_x = nn.Linear(in_channels, out_channels)
        self.linear_gate = nn.Linear(in_channels, out_channels)
        self._init_weights(self.linear_x)
        self._init_weights(self.linear_gate)

        if activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "prelu":
            self.act = nn.PReLU()

    def _init_weights(self, layer):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)

    def forward(self, cls_token, reg_token):
        cls_token = self.act(self.linear_x(cls_token))
        reg_token = self.linear_gate(reg_token)
        return cls_token * reg_token


class SE_Block(nn.Module):
    def __init__(self, in_channels, out_channels, in_num=1, out_num=50):
        super(SE_Block, self).__init__()

        self.glus = nn.ModuleList([GLU(in_channels, in_channels) for _ in range(4)])
        self.cls_linear = nn.Sequential(
            Rearrange('b l c -> b c l'),
            nn.Linear(in_num, out_num),
            Rearrange('b c l -> b l c')
        )
        self.out_linear = nn.Linear(in_channels, out_channels)
        self.out_num = out_num

    def _init_weights(self, layer):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)

    def forward(self, cls_tokens, reg_tokens):
        """
        :param cls_tokens: [batch_size, 1, C]
        :param reg_tokens: [batch_size, 4, C]
        """
        if cls_tokens.dim() == 2:
            cls_tokens = cls_tokens.unsqueeze(1)   # [B, 1, C]

        glu_outputs = [self.glus[i](cls_tokens, reg_tokens[:, i:i + 1, :]) for i in range(reg_tokens.size(1))]

        cls_token_expanded = self.cls_linear(cls_tokens)   # [B, 100, C]

        result = []
        step = self.out_num // reg_tokens.size(1)
        for i in range(reg_tokens.size(1)):
            result.append(cls_token_expanded[:, i * step:(i + 1) * step, :] * glu_outputs[i])

        result = torch.cat(result, dim=1)
        return self.out_linear(result)


class BinaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BinaryClassifier, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        out = self.pooling(x.permute(0, -1, 1)).view(x.size(0), -1)
        return self.fc(out).squeeze(-1)


def test_glu():
    glu = GLU(512, 512)
    cls_token = torch.randn(4, 1, 512)
    reg_token = torch.randn(4, 1, 512)
    result = glu(cls_token, reg_token)
    print(result.shape)


def test_seblock():
    block = SE_Block(512, 512)
    cls_tokens = torch.randn(4, 1, 512)
    reg_tokens = torch.randn(4, 4, 512)
    result = block(cls_tokens, reg_tokens)
    print(result.shape)


def test_BinaryClassifier():
    classifier = BinaryClassifier(256, 256, 2)
    x = torch.randn([4, 200, 256])
    y = classifier(x)
    print(y.shape)


if __name__ == '__main__':
    test_BinaryClassifier()