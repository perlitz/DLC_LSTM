from yacs.config import CfgNode as CN
import torch

_C = CN()

_C.SYSTEM = CN()
_C.SYSTEM.SEED = 1111
_C.SYSTEM.MODEL_SAVE_PATH = '/home/yotampe/Code/Edu/DLC_LSTM/model/saved_models/model.pth'
_C.SYSTEM.MODEL_LOAD_PATH = False
# # Number of GPUS to use in the experiment
# _C.SYSTEM.NUM_GPUS = 8
# # Number of workers for doing things
# _C.SYSTEM.NUM_WORKERS = 4

_C.TRAIN = CN()
_C.TRAIN.DATA_PATH = '/home/yotampe/Code/Edu/DLC_LSTM/data/PTB/ptb.'
_C.TRAIN.BATCH_SIZE = 20
_C.TRAIN.EVAL_BATCH_SIZE = 20
_C.TRAIN.SEQ_LEN = 20
_C.TRAIN.CLIP = 5.
_C.TRAIN.LOG_INTERVAL = 200
_C.TRAIN.N_EPOCHS = 20
_C.TRAIN.INIT_LR = 1
_C.TRAIN.DROP_LR_AFTER = 6
_C.TRAIN.DROP_LR_BY = 1.2
_C.TRAIN.WIN_INIT = 0.05
# # A very important hyperparameter
# _C.TRAIN.HYPERPARAMETER_1 = 0.1
# # The all important scales for the stuff
# _C.TRAIN.SCALES = (2, 4, 8, 16)
_C.NET = CN()
_C.NET.EMBED_DIM = 200
_C.NET.HIDDEN_DIM = 200
_C.NET.N_LAYERS = 2
_C.NET.DROP_PROB = 0.5
# def get_cfg_defaults():
#   """Get a yacs CfgNode object with default values for my_project."""
#   # Return a clone so that the defaults will not be altered
#   # This is for the "local variable" use pattern
#   return _C.clone()

cfg = _C