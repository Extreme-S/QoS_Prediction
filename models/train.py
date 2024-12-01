import os, sys, torch
import numpy as np

sys.path.append("/Users/lyw/projects/ECNU/QoS_Prediction")

from utils.dataloader import WSDREAM_1_MatrixDataset, WSDREAM_1_InfoDataset, ToTorchDataset
from root import absolute
from torch import nn, optim
from torch.nn.modules import loss
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils.evaluation import mae, mse, rmse
from root import ROOT
from utils.loss import CauchyLoss
from const import *
from utils.model_util import freeze_random
from utils.mylogger import myLog
from utils.data_process import *
from models.model import *

para = {
    "dataType": "rt",  # set the dataType as 'rt' or 'tp'
    "dataPath": "data/dataset#1/",
    "outPath": "result/",
    "metrics": ["MAE", "RMSE"],  # delete where appropriate
    "density": [0.025, 0.05, 0.075, 0.10, 0.15, 0.20],  # matrix density
    "epochs": 100,
    "lr": 2e-4,
    "batchSize": 64,
    "topK": 10,  # the parameter of TopK similar users or services, the default
    "dimension": 32,  # dimenisionality of the latent factors
    "etaInit": 0.01,  # inital learning rate. We use line search
    "lambda": 30,  # L2 regularization parameter
    "alpha": 0.4,  # the parameter of combination, 0.4 as in the reference paper
    "maxIter": 300,  # the max iterations
    "saveTimeInfo": False,  # whether to keep track of the running time
    "saveLog": True,  # whether to save log into file
    "debugMode": False,  # whether to record the debug info
    "parallelMode": False,  # whether to leverage multiprocessing for speedup
}

rt_data = WSDREAM_1_MatrixDataset(para["dataType"])
user_data = WSDREAM_1_InfoDataset("user", ["[Country]", "[AS]"])
item_data = WSDREAM_1_InfoDataset("service", ["[WSDL Address]", "[Service Provider]", "[Country]", "[AS]"])
adj_matrix = build_adj_matrix(para["dataPath"])  # 构建GCN的输入邻接矩阵


for density in para["density"]:

    train_data, test_data = rt_data.split_train_test(density)
    train_dataloader = DataLoader(ToTorchDataset(train_data), batch_size=para["batchSize"], drop_last=False)
    test_dataloader = DataLoader(ToTorchDataset(test_data), batch_size=para["batchSize"], drop_last=False)

    loss = nn.L1Loss()
    model = GlbpModel(loss, rt_data, adj_matrix, user_data, item_data, latent_dim=para["dimension"])
    optimizer = Adam(model.parameters(), lr=para["lr"], weight_decay=2e-4)
    model.fit(train_dataloader, para["epochs"], optimizer, eval_loader=test_dataloader, save_filename=f"Density_{density}")

    y, y_pred = model.predict(test_dataloader, True)
    mae_ = mae(y, y_pred)
    rmse_ = rmse(y, y_pred)
    model.logger.info(f"Density:{density:.2f}, mae:{mae_:.4f}, rmse:{rmse_:.4f}")
