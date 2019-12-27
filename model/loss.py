import torch

def log_softmax(output):
    return torch.exp(output) / torch.sum(torch.exp(output), axis = 0, keepdim=True)

def nlll(output_probs, target):
    batch_size   = output_probs.size(1) # since batch is not firs
    output_probs = output_probs.view(-1, output_probs.size(2))
    target_probs = output_probs[range(len(target.reshape(-1))), target.reshape(-1)]
    return torch.mean(-torch.log(target_probs), dim=0) * batch_size

def perplexity(loss_list):
    return torch.exp(torch.mean(loss_list)) # I