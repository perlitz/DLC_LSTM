import numpy as np
from string import punctuation
from data.data import Corpus, batchify, get_batch
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
from model.model import LangModelRNN
from config import cfg
from model.loss import perplexity
from train import train_epoch, evaluate, save_model_if_better
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboardX import SummaryWriter

# from sacred import Experiment
# from sacred.observers import FileStorageObserver
#
# ex = Experiment()
# output_dir = '/home/yotampe/Code/Edu/DLC_LSTM/runs/sacred_exps'
# ex.observers.append(FileStorageObserver.create(output_dir))
#
# @ex.automain
def main():#_run):

    cfg.merge_from_file('lstm_with_drop_mac.yaml')
    cfg.freeze()
    print(cfg)

    torch.manual_seed(cfg.SYSTEM.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # GET DATA
    corpus = Corpus(cfg.TRAIN.DATA_PATH)
    train_data = batchify(corpus.train, cfg.TRAIN.BATCH_SIZE, device)[:100]
    valid_data = batchify(corpus.valid, cfg.TRAIN.EVAL_BATCH_SIZE, device)[:100]
    test_data = batchify(corpus.test, cfg.TRAIN.EVAL_BATCH_SIZE, device)
    n_tokens = len(corpus.dictionary)

    #################################################
    #################################################
    def load_glove_embeddings(path, word2idx, embedding_dim):
        with open(path, encoding='utf-8') as f:
            embeddings = np.zeros((len(word2idx), embedding_dim))
            for line in f.readlines():
                values = line.split()
                word = values[0]
                index = word2idx.get(word)
                if index:
                    vector = np.array(values[1:], dtype='float32')
                    embeddings[index] = vector
            return torch.from_numpy(embeddings).float()

    if False:
        glove_path = '/home/yotampe/Code/Edu/glove.6B.200d.txt'
        glove = load_glove_embeddings(glove_path, corpus.dictionary.word2idx, 200)
    else:
        glove = None
    #################################################
    #################################################

    # BUILD MODEL


    if cfg.SYSTEM.MODEL_LOAD_PATH:
        with open(cfg.SYSTEM.LOAD_MODEL_PATH, 'rb') as f:
            model, criterion, optimizer = torch.load(f)
    else:
        model = LangModelRNN(n_tokens = n_tokens,
                             embedding_dim = cfg.NET.EMBED_DIM,
                             hidden_dim = cfg.NET.HIDDEN_DIM,
                             n_layers = cfg.NET.N_LAYERS,
                             dropout = cfg.NET.DROP_PROB,
                             rnn_type='LSTM',
                             glove=glove).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(lr=cfg.TRAIN.INIT_LR, params=model.parameters())

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=(1/1.2), verbose=True)

    train_loss_list = []
    valid_loss_list  = []
    writer = SummaryWriter()
    # fig, ax = plt.subplots()
    # plt.ion()

    # TRAIN ONE EPOCH
    for epoch in range(1,cfg.TRAIN.N_EPOCHS+1):

        train_loss_list.append(train_epoch(n_tokens, criterion, epoch, model, optimizer, train_data.to(device)))
        valid_loss_list.append(evaluate(n_tokens ,model, valid_data.to(device), criterion, epoch))
        save_model_if_better(valid_loss_list ,model, optimizer, criterion)
        # scheduler.step(epoch)
        if epoch > cfg.TRAIN.DROP_LR_AFTER:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= cfg.TRAIN.DROP_LR_BY
                print(f'Reduced LR by a factor of {cfg.TRAIN.DROP_LR_BY}')

        writer.add_scalar('train_loss', train_loss_list[-1], epoch)
        writer.add_scalar('test_loss',   valid_loss_list[-1], epoch)

    plt.plot(train_loss_list, label = 'Train loss')
    plt.plot(valid_loss_list, label = 'Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # ex.log_scalar('Valid loss', valid_loss_list[-1])
    # ex.log_scalar('Train loss', train_loss_list[-1])

    test_loss = evaluate(n_tokens ,model, test_data.to(device), criterion, 999)
    # ex.log_scalar('Test loss', test_loss)

    print('=' * 89)
    print('| End of training | test loss {:5.2f}'.format(test_loss))
    print('=' * 89)
# # read data from text files
if __name__ == '__main__':
    main()

