"""Masked Language Model Layer"""
import torch
import torch.nn as nn


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden: int, vocab_size: int):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.transform = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.LayerNorm(hidden, eps=1e-12))
        self.decoder = nn.Linear(hidden, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, x):
        """
        :param x: (batch_size, max_len, hidden_size)
        :return: (batch_size, max_len, vocab_size)
        """
        hidden = self.transform(x)
        return self.decoder(hidden) + self.bias
