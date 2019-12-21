import torch

def perplexity(output, target, cross_entropy_loss):
    return torch.exp(cross_entropy_loss(output, target))