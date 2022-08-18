import torch
from torch.utils.data import DataLoader
from timeit import default_timer

from models import *
from utils import *

class NSEModel:
    def __init__(self, args, shape, data):
        self.logs = dict()
        self.logs['train_loss']=[]
        self.logs['train_loss_f_t_rec']=[]
        self.logs['train_loss_u_t_rec']=[]
        self.logs['train_loss_trans']=[]
        self.logs['train_loss_trans_latent']=[]
        self.logs['test_loss']=[]
        self.logs['test_loss_f_t_rec']=[]
        self.logs['test_loss_u_t_rec']=[]
        self.logs['test_loss_trans']=[]
        self.logs['test_loss_trans_latent']=[]
        self.params = args
        self.device = torch.device('cuda:{}'.format(self.params.gpu) if torch.cuda.is_available() else 'cpu')

        model_params = dict()
        model_params['modes'] = self.params.modes
        model_params['width'] = self.params.width
        model_params['L'] = self.params.L
        f_channels = self.params.f_channels

        self.epochs = self.params.epochs
        self.batch_size = self.params.batch_size
        lr = self.params.lr
        wd = self.params.wd
        step_size = self.params.step_size
        gamma = self.params.gamma

        train_data, test_data = data.trans2Dataset()
        self.train_loader = DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=test_data, batch_size=self.batch_size, shuffle=False)

        self.model = FNO_ensemble(model_params, shape, f_channels=f_channels).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

    def cout_params(self):
        c = 0
        for p in list(self.model.parameters()):
            c += reduce(operator.mul, list(p.size()))
        return c

    def print_params(self):
        print(f'L: {self.params.L}, modes: {self.params.modes}, width: {self.params.width}')
        print(f'epochs: {self.epochs}, batch_size: {self.batch_size}, lr: {self.params.lr}, step_size: {self.params.step_size}, gamma: {self.params.gamma}')
        print(f'lambda: {self.params.lambda1}, {self.params.lambda2}, {self.params.lambda3}, {self.params.lambda4}, f_channels: {self.params.f_channels}')
    
    def train_test(self, epoch):
        lambda1, lambda2, lambda3, lambda4 = self.params.lambda1, self.params.lambda2, \
                                             self.params.lambda3, self.params.lambda4
        self.model.train()

        t1 = default_timer()
        train_loss = AverageMeter()
        train_loss1 = AverageMeter()
        train_loss2 = AverageMeter()
        train_loss3 = AverageMeter()
        train_loss4 = AverageMeter()

        device = self.device
        for x_train, y_train in self.train_loader:
            x_train, y_train = x_train.to(device), y_train.to(device)
            
            self.optimizer.zero_grad()

            # split data read in train_loader
            in_train, f_train = x_train[:, :, :, :-1], x_train[:, 0, 0, -1]
            out_train, Cd_train, Cl_train = y_train[:, :, :, :-2], y_train[:, 0, 0, -2], y_train[:, 0, 0, -1]
            # put data into model
            pred, x_rec, f_rec, trans_out = self.model(in_train, f_train)
            out_latent = self.model.stat_en(out_train)
            in_rec = x_rec[:, :, :, :3]
            # prediction items
            out_pred = pred[:, :, :, :3]
            Cd_pred = torch.mean(pred[:, :, :, -2].reshape(self.batch_size, -1), 1)
            Cl_pred = torch.mean(pred[:, :, :, -1].reshape(self.batch_size, -1), 1)

            # loss1: prediction loss; loss2: rec loss of state
            # loss3: rec loss of f; loss4: latent loss
            loss1 = rel_error(out_pred, out_train).mean()\
                    + rel_error(Cd_pred, Cd_train).mean()\
                    + rel_error(Cl_pred, Cl_train).mean()
            loss2 = rel_error(in_rec, in_train).mean()
            loss3 = rel_error(f_rec, f_train).mean()
            # loss3 = F.mse_loss(f_rec, f_train, reduction='mean')
            loss4 = rel_error(trans_out, out_latent).mean()
            loss = lambda1 * loss1 + lambda2 * loss2 + lambda3 * loss3 + lambda4 * loss4
            
            loss.backward()
            self.optimizer.step()

            train_loss.update(loss.item(), x_train.shape[0])
            train_loss1.update(loss1.item(), x_train.shape[0])
            train_loss2.update(loss2.item(), x_train.shape[0])
            train_loss3.update(loss3.item(), x_train.shape[0])
            train_loss4.update(loss4.item(), x_train.shape[0])
        
        self.logs['train_loss'].append(train_loss.avg)
        self.logs['train_loss_f_t_rec'].append(train_loss3.avg)
        self.logs['train_loss_u_t_rec'].append(train_loss2.avg)
        self.logs['train_loss_trans'].append(train_loss1.avg)
        self.logs['train_loss_trans_latent'].append(train_loss4.avg)
        
        self.scheduler.step()
        t2 = default_timer()
        
        self.model.eval()

        test_loss = AverageMeter()
        test_loss1 = AverageMeter()
        test_loss2 = AverageMeter()
        test_loss3 = AverageMeter()
        test_loss4 = AverageMeter()

        with torch.no_grad():
            for x_test, y_test in self.test_loader:
                x_test, y_test = x_test.to(device), y_test.to(device)

                # split data read in test_loader
                in_test, f_test = x_test[:, :, :, :-1], x_test[:, 0, 0, -1]
                out_test, Cd_test, Cl_test = y_test[:, :, :, :-2], y_test[:, 0, 0, -2], y_test[:, 0, 0, -1]
                # put data into model
                pred, x_rec, f_rec, trans_out = self.model(in_test, f_test)
                out_latent = self.model.stat_en(out_test)
                in_rec = x_rec[:, :, :, :3]
                # prediction items
                out_pred = pred[:, :, :, :3]
                Cd_pred = torch.mean(pred[:, :, :, -2].reshape(self.batch_size, -1), 1)
                Cl_pred = torch.mean(pred[:, :, :, -1].reshape(self.batch_size, -1), 1)
                loss1 = rel_error(out_pred, out_test).mean()\
                        + rel_error(Cd_pred, Cd_test).mean()\
                        + rel_error(Cl_pred, Cl_test).mean()
                loss2 = rel_error(in_rec, in_test).mean()
                loss3 = rel_error(f_rec, f_test).mean()
                loss4 = rel_error(trans_out, out_latent).mean()
                loss = lambda1 * loss1 + lambda2 * loss2 + lambda3 * loss3 + lambda4 * loss4

                test_loss.update(loss.item(), x_test.shape[0])
                test_loss1.update(loss1.item(), x_test.shape[0])
                test_loss2.update(loss2.item(), x_test.shape[0])
                test_loss3.update(loss3.item(), x_test.shape[0])
                test_loss4.update(loss4.item(), x_test.shape[0])
            
            self.logs['test_loss'].append(test_loss.avg)
            self.logs['test_loss_f_t_rec'].append(test_loss3.avg)
            self.logs['test_loss_u_t_rec'].append(test_loss2.avg)
            self.logs['test_loss_trans'].append(test_loss1.avg)
            self.logs['test_loss_trans_latent'].append(test_loss4.avg)
        
        print('# {} {:1.3f} | loss1: {:1.2e}  loss2: {:1.2e}  loss3: {:1.2e} loss4: {:1.2e} | loss1: {:1.2e} loss2: {:1.2e}  loss3: {:1.2e} loss4: {:1.2e}'
              .format(epoch, t2-t1, train_loss1.avg, train_loss2.avg, train_loss3.avg, train_loss4.avg, test_loss1.avg, test_loss2.avg, test_loss3.avg, test_loss4.avg))

    def process(self):
        for epoch in range(1, self.epochs+1):
            self.train_test(epoch)

    def get_logs(self):
        return self.logs


class NSECtr:
    def __init__(self, args, operator_path, shape):
        self.device = torch.device('cuda:{}'.format(self.params.gpu) if torch.cuda.is_available() else 'cpu')
        
        state_dict, logs = torch.load(operator_path)
        L = logs['args'].L
        modes = logs['args'].modes
        width = logs['args'].width
        model_params = dict()
        model_params['modes'] = modes
        model_params['width'] = width
        model_params['L'] = L
        f_channels = logs['args'].f_channels

        self.model = FNO_ensemble(model_params, shape, f_channels).to(self.device)
        self.model.eval()