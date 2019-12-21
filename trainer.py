import numpy as np
from string import punctuation
from data.data import Corpus, batchify, get_batch
from torch.utils.data import TensorDataset, DataLoader
import torch
from model.model import LangModelRNN
from config import cfg
from model.loss import perplexity
from train import train_epoch, evaluate
import time
import matplotlib.pyplot as plt
import seaborn as sns
def main():

    cfg.merge_from_file("experiment.yaml")
    cfg.freeze()
    print(cfg)

    torch.manual_seed(cfg.SYSTEM.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # GET DATA
    corpus = Corpus(cfg.TRAIN.DATA_PATH)
    train_data = batchify(corpus.train, cfg.TRAIN.BATCH_SIZE, device)
    val_data = batchify(corpus.valid, cfg.TRAIN.BATCH_SIZE, device)
    test_data = batchify(corpus.test, cfg.TRAIN.EVAL_BATCH_SIZE, device)
    n_tokens = len(corpus.dictionary)
    # BUILD MODEL
    model = LangModelRNN(n_tokens = n_tokens,
                               embedding_dim = cfg.NET.EMBED_DIM,
                               hidden_dim = cfg.NET.HIDDEN_DIM,
                               n_layers = cfg.NET.N_LAYERS,
                               dropout = cfg.NET.DROP_PROB,
                               rnn_type='LSTM')

    criterion = torch.nn.CrossEntropyLoss()
    lr = 10
    optimizer = torch.optim.SGD(lr=lr, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2, last_epoch=-1)
    train_loss_list = []
    test_loss_list  = []

    fig, ax = plt.subplots()

    # TRAIN ONE EPOCH
    for epoch in range(cfg.TRAIN.N_EPOCHS):

        train_loss_list.append(train_epoch(n_tokens, criterion, epoch, model, optimizer, train_data))
        test_loss_list.append(evaluate(n_tokens ,model, test_data, criterion, epoch))
        scheduler.step(epoch)
        ax.plot(train_loss_list, range(epoch))
        ax.plot(test_loss_list,  range(epoch))
        fig.show()









# read data from text files
if __name__ == '__main__':
    main()

