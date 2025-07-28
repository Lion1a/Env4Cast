from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Env4Cast
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
import math
from datetime import datetime


import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.adj_haversine = None  # 初始化 adj_haversine 属性为 None
        self.weight = 1



    def _build_model(self):
        model_dict = {
            'Env4Cast': Env4Cast,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.L1Loss()
        return criterion

    def get_reciprocal_laplacian(self,laplacian_diff):
        """
        处理三维拉普拉斯矩阵的倒数
        """
        epsilon = 1e-12
        # 添加微小常数避免除零
        protected_matrix = laplacian_diff + epsilon
        # 计算倒数并处理符号（拉普拉斯矩阵可能有负值）
        reciprocal = np.where(np.abs(laplacian_diff) > epsilon,
                              1.0 / protected_matrix,
                              0)
        return reciprocal

    def diffgraph(self):
        if self.adj_haversine is not None:
            return self.adj_haversine, self.laplacian_diff, self.laplacian_heat
        # 加载数据
        self.adj_haversine = np.load('/data/data_1/zxy/pathformer-main/data-preparation/adj_haversine_20000_22000.npy')
        self.g_diff = self.get_reciprocal_laplacian(self.adj_haversine)
        degree_matrix_diff = np.diag(np.sum(self.g_diff, axis=1))
        laplacian_diff = degree_matrix_diff-self.g_diff
        sigma = np.mean(self.adj_haversine[self.adj_haversine > 0])

        G_heat = np.exp(- (self.adj_haversine ** 2) / (sigma ** 2))

        # G_heat = np.exp(- (self.adj_haversine ** 2))
        degree_matrix_heat = np.diag(np.sum(G_heat, axis=1))

        laplacian_heat = degree_matrix_heat - G_heat
        return self.adj_haversine, laplacian_diff, laplacian_heat

    # def diffgraph(self):
    #     if self.adj_haversine is not None:
    #         return self.adj_haversine, self.laplacian_diff, self.laplacian_heat
    #     # 加载数据
    #     self.adj_haversine = np.load('/data/data_1/zxy/Env4Cast-main/data-preparation/adj_haversine_20000_22000.npy')
    #
    #     degree_matrix = np.diag(np.sum(self.adj_haversine, axis=1))
    #     # 计算拉普拉斯矩阵 L = D - A
    #     laplacian_diff = degree_matrix - self.adj_haversine
    #
    #     G_heat = np.exp(- (self.adj_haversine ** 2))
    #     laplacian_heat = degree_matrix - G_heat
    #     return self.adj_haversine, laplacian_diff, laplacian_heat

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        # self.adj_haversine, self.laplacian_diff, self.laplacian_heat = self.diffgraph()
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,batch_velo) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_velo = batch_velo.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)


                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model=='Env4Cast':
                            outputs, balance_loss, contrastive_loss = self.model(batch_x,batch_velo,self.adj_haversine, self.laplacian_diff, self.laplacian_heat)
                        else:
                            outputs = self.model(batch_x,batch_velo,self.adj_haversine)

                else:
                    if self.args.model=='Env4Cast':
                        outputs, balance_loss, contrastive_loss = self.model(batch_x,batch_velo,self.adj_haversine, self.laplacian_diff, self.laplacian_heat)
                    else:
                        outputs = self.model(batch_x,batch_velo,self.adj_haversine)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):

        os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'

        self.adj_haversine, self.laplacian_diff, self.laplacian_heat = self.diffgraph()

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        total_num = sum(p.numel() for p in self.model.parameters())
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_velo) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_velo = batch_velo.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)



                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model=='Env4Cast':
                            outputs, balance_loss, contrastive_loss = self.model(batch_x,batch_velo,self.adj_haversine, self.laplacian_diff, self.laplacian_heat)
                        else:
                            outputs = self.model(batch_x,batch_velo)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.model == 'Env4Cast':
                        outputs, balance_loss, contrastive_loss = self.model(batch_x,batch_velo,self.adj_haversine, self.laplacian_diff, self.laplacian_heat)
                    else:
                        outputs = self.model(batch_x,batch_velo)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    if self.args.model=="Env4Cast":
                        loss = loss + balance_loss + self.weight*contrastive_loss

                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model


    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        # self.adj_haversine, self.laplacian_diff, self.laplacian_heat = self.diffgraph()
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,batch_velo) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_velo = batch_velo.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)


                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model=='Env4Cast':
                            outputs, balance_loss, contrastive_loss = self.model(batch_x,batch_velo,self.adj_haversine, self.laplacian_diff, self.laplacian_heat)
                        else:
                            outputs = self.model(batch_x,batch_velo,self.adj_haversine)
                else:
                    if self.args.model == 'Env4Cast':
                        outputs, balance_loss,contrastive_loss = self.model(batch_x,batch_velo,self.adj_haversine, self.laplacian_diff, self.laplacian_heat)
                    else:
                        outputs = self.model(batch_x,batch_velo)
                f_dim = -1 if self.args.features == 'MS' else 0

                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        mae, mse, rmse, mape, smape, mspe, rse, corr = metric(preds, trues)
        print('rmse:{}, mse:{}, mae:{}, rse:{}, mape:{}, smape:{}'.format(rmse, mse, mae, rse, mape, smape))
        f = open("result_6_contrastive.txt", 'a')
        f.write(setting + "  \n")
        f.write('rmse:{}, mse:{}, mae:{}, rse:{}, mape:{}, smape:{}'.format(rmse, mse, mae, rse, mape, smape))
        f.write('\n')
        f.write('\n')
        timestampp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # 保存结果文件，加上时间戳
        np.save(os.path.join(folder_path, f'preds_{timestampp}_32000_34000_3.npy'), preds)
        np.save(os.path.join(folder_path, f'trues_{timestampp}_32000_34000_3.npy'), trues)
        f.close()
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        adj_haversine=self.diffgraph()
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))


        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,batch_velo) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_velo = batch_velo.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)


                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model=='Env4Cast':
                            outputs, a_loss, contrastive_loss = self.model(batch_x,batch_velo,self.adj_haversine, self.laplacian_diff, self.laplacian_heat)
                        else:
                            outputs = self.model(batch_x,batch_velo,self.adj_haversine)

                else:
                    if self.args.model == 'Env4Cast':
                        outputs, a_loss, contrastive_loss = self.model(batch_x,batch_velo,self.adj_haversine, self.laplacian_diff, self.laplacian_heat)
                    else:
                        outputs = self.model(batch_x,batch_velo,self.adj_haversine)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)
        #
        # np.save(folder_path + 'real_prediction.npy', preds)

        return



    # Haversine 距离计算函数
    def haversine(self,lat1, lon1, lat2, lon2):
        R = 6371  # 地球半径，单位 km
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c  # 返回距离，单位 km

    def build_adjacency_matrix(self,longitudes, latitudes):
        n = len(longitudes)
        adj_matrix = np.zeros((n, n))  # 初始化邻接矩阵

        for i in range(n):
            for j in range(i + 1, n):
                lat1, lon1 = latitudes[i], longitudes[i]
                lat2, lon2 = latitudes[j], longitudes[j]

                distance = self.haversine(lat1, lon1, lat2, lon2)

                # 将计算出的距离存入矩阵
                adj_matrix[i, j] = adj_matrix[j, i] = distance  # 因为是无向图，填充对称位置

        return adj_matrix