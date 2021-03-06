import time
from config import cfg
import torch
import torch.nn as nn
from data.data import get_batch
from model.loss import perplexity
import numpy as np


def train_epoch(n_tokens, criterion, epoch, model, optimizer, train_data):

    model.train()
    total_loss = 0.
    cur_loss = 0.
    hidden = model.init_hidden(cfg.TRAIN.BATCH_SIZE)
    start_time = time.time()
    for batch, seq_num in enumerate(range(0, train_data.size(0) - 1, cfg.TRAIN.SEQ_LEN)):
        data, targets = get_batch(train_data, seq_num, cfg.TRAIN.SEQ_LEN)
        model.zero_grad()
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)

        loss = criterion(output.view(-1, n_tokens), targets.view(-1))
        # loss = perplexity(output.view(-1, n_tokens), targets, criterion)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), cfg.TRAIN.CLIP)
        for p in model.parameters():
            p.data.add_(-get_lr(optimizer), p.grad.data)


        optimizer.step()
        total_loss += loss.item()

        if batch % cfg.TRAIN.LOG_INTERVAL == 0 and batch > 0:
            cur_loss = total_loss / cfg.TRAIN.LOG_INTERVAL
            elapsed = time.time() - start_time
            print('train : | epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f}'.format(
                epoch, batch, len(train_data) // cfg.TRAIN.SEQ_LEN, get_lr(optimizer),
                              elapsed * 1000 / cfg.TRAIN.LOG_INTERVAL, cur_loss))
            total_loss = 0
            start_time = time.time()

    return np.mean(cur_loss)


def evaluate(n_tokens ,model, test_data, criterion, epoch):
    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(cfg.TRAIN.EVAL_BATCH_SIZE)
    with torch.no_grad():

        for batch, seq_num in enumerate(range(0, test_data.size(0) - 1, cfg.TRAIN.SEQ_LEN)):
            data, targets = get_batch(test_data, seq_num, cfg.TRAIN.SEQ_LEN)
            output, hidden = model(data, hidden)
            hidden = repackage_hidden(hidden)
            loss = perplexity(output.view(-1, n_tokens), targets, criterion)
            total_loss += len(data) * loss

    print('evaluate : | epoch {:3d} | loss {:5.2f}'.format(
                        epoch,         total_loss / (len(test_data) - 1)))

    return total_loss.item() / (len(test_data) - 1)

def save_model_if_better(eval_loss_list, model, criterion, optimizer):
    if cfg.SYSTEM.MODEL_SAVE_PATH and (len(eval_loss_list) > 1):
        if eval_loss_list[-1] < min(eval_loss_list[:-1]):
            with open(cfg.SYSTEM.MODEL_SAVE_PATH, 'wb+') as f:
                torch.save([model, criterion, optimizer], f)
                print('Saving model (new best validation)')



def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

