import torch

def log_softmax(output):
    return torch.log(torch.exp(output) / torch.sum(torch.exp(output), dim = 0, keepdim=True))

def nlll(output_logs, targets, batch_size):

    answerprobs = output_logs[range(len(targets.view(-1))), targets.view(-1)]
    return -torch.mean(answerprobs * batch_size)


# def perplexity(loss_list):
#     return torch.exp(torch.mean(loss_list)) # I
def perplexity(output, target, cross_entropy_loss):
    return torch.exp(cross_entropy_loss(output, target.view(-1)))
