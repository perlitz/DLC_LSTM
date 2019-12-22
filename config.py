from yacs.config import CfgNode as CN
import torch

_C = CN()

_C.SYSTEM = CN()
_C.SYSTEM.SEED = 1111
_C.SYSTEM.MODEL_SAVE_PATH = '/Users/yotam/Documents/DLC_LSTM/saved/models/model.pt'
_C.SYSTEM.MODEL_LOAD_PATH = False
# # Number of GPUS to use in the experiment
# _C.SYSTEM.NUM_GPUS = 8
# # Number of workers for doing things
# _C.SYSTEM.NUM_WORKERS = 4

_C.TRAIN = CN()
_C.TRAIN.DATA_PATH = '/Users/yotam/Documents/DLC_LSTM/data/PTB/ptb.'
_C.TRAIN.EPOCHS = 2
_C.TRAIN.BATCH_SIZE = 20
_C.TRAIN.EVAL_BATCH_SIZE = 10
_C.TRAIN.SEQ_LEN = 35
_C.TRAIN.CLIP = 0.25
_C.TRAIN.LOG_INTERVAL = 200
_C.TRAIN.N_EPOCHS = 5
_C.TRAIN.INIT_LR = 10
# # A very important hyperparameter
# _C.TRAIN.HYPERPARAMETER_1 = 0.1
# # The all important scales for the stuff
# _C.TRAIN.SCALES = (2, 4, 8, 16)
_C.NET = CN()
_C.NET.EMBED_DIM = 400
_C.NET.HIDDEN_DIM = 200
_C.NET.N_LAYERS = 2
_C.NET.DROP_PROB = 0.5
# def get_cfg_defaults():
#   """Get a yacs CfgNode object with default values for my_project."""
#   # Return a clone so that the defaults will not be altered
#   # This is for the "local variable" use pattern
#   return _C.clone()

cfg = _C