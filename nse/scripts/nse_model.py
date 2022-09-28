import torch
from torch.utils.data import DataLoader
from timeit import default_timer
import gc

from scripts.models import *
from scripts.utils import *

class NSEModel_FNO:
    def __init__(self, args, shape, dt):
        self.params = args
        self.dt = dt
        self.device = torch.device('cuda:{}'.format(self.params.gpu) if torch.cuda.is_available() else 'cpu')

        model_params = dict()
        model_params['modes'] = self.params.modes
        model_params['width'] = self.params.width
        model_params['L'] = self.params.L
        model_params['shape'] = shape
        model_params['f_channels'] = self.params.f_channels
        
        self.pred_model = FNO_ensemble(model_params).to(self.device)
        self.phys_model = state_mo(model_params).to(self.device)
        self.pred_optimizer = torch.optim.Adam(self.pred_model.parameters(), lr=self.params.lr, weight_decay=self.params.wd)
        self.phys_optimizer = torch.optim.Adam(self.phys_model.parameters(), lr=self.params.lr * 5, weight_decay=self.params.wd)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.pred_optimizer, step_size=self.params.step_size, gamma=self.params.gamma)

    def count_params(self):
        c = 0
        for p in list(self.pred_model.parameters()):
            c += reduce(operator.mul, list(p.size()))
        for p in list(self.phys_model.parameters()):
            c += reduce(operator.mul, list(p.size()))
        return c
    
    def data_train_test(self, epoch, train_loader, test_loader, logs):
        lambda1, lambda2, lambda3, lambda4, lambda5 = self.params.lambda1, self.params.lambda2, \
                                                      self.params.lambda3, self.params.lambda4, self.params.lambda5
        self.pred_model.train()
        self.phys_model.train()

        t1 = default_timer()
        train_log = PredLog(mode='train', length=self.params.batch_size)
        test_log = PredLog(mode='test', length=self.params.batch_size)

        device = self.device
        for x_train, y_train in train_loader:
            x_train, y_train = x_train.to(device), y_train.to(device)
            
            self.pred_optimizer.zero_grad()
            self.phys_optimizer.zero_grad()

            # split data read in train_loader
            in_train, ctr_train = x_train[:, :, :, :-1], x_train[:, 0, 0, -1]
            out_train, Cd_train, Cl_train = y_train[:, :, :, :-2], y_train[:, 0, 0, -2], y_train[:, 0, 0, -1]
            opt_train = [out_train, Cd_train, Cl_train]

            # put data to generate 4 loss
            loss1, loss2, loss3, loss4 = self.pred_loss(in_train, ctr_train, opt_train)
            # physical loss
            mod = self.phys_model(in_train, ctr_train, out_train)
            loss_pde = ((Lpde(out_train, in_train, self.dt) + mod) ** 2).mean()
    
            loss_pred = lambda1 * loss1 + lambda2 * loss2 + lambda3 * loss3 + lambda4 * loss4 #+ lambda5 * loss_pde
            loss_pred.backward()
            loss_pde.backward()
            self.pred_optimizer.step()
            self.phys_optimizer.step()

            train_log.update(loss1, loss2, loss3, loss4, loss_pde)
        
        self.scheduler.step()
        t2 = default_timer()
        train_log.save_log(logs)
        self.pred_model.eval()
        self.phys_model.eval()

        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test, y_test = x_test.to(device), y_test.to(device)
                # split data read in test_loader
                in_test, ctr_test = x_test[:, :, :, :-1], x_test[:, 0, 0, -1]
                out_test, Cd_test, Cl_test = y_test[:, :, :, :-2], y_test[:, 0, 0, -2], y_test[:, 0, 0, -1]
                opt_test = [out_test, Cd_test, Cl_test]
                loss1, loss2, loss3, loss4 = self.pred_loss(in_test, ctr_test, opt_test)
                mod = self.phys_model(in_test, ctr_test, out_test)
                loss_pde = ((Lpde(out_test, in_test, self.dt) + mod) ** 2).mean()
                test_log.update(loss1, loss2, loss3, loss4, loss_pde)
            test_log.save_log(logs)

        print('# {} {:1.2f} | (pred): {:1.2e}  (rec)state: {:1.2e}  ctr: {:1.2e} (latent): {:1.2e} (pde): {:1.2e} |'
              .format(epoch, t2-t1, train_log.loss1.avg, train_log.loss2.avg, train_log.loss3.avg, train_log.loss4.avg, train_log.loss_pde.avg) + 
              '(pred): {:1.2e}  (rec)state: {:1.2e}  ctr: {:1.2e} (latent): {:1.2e} (pde): {:1.2e}'
              .format(test_log.loss1.avg, test_log.loss2.avg, test_log.loss3.avg, test_log.loss4.avg, test_log.loss_pde.avg))

    def phys_train(self, phys_epoch, train_loader):
        loss_pde = AverageMeter()
        t3 = default_timer()

        for x_train, _ in train_loader:
            x_train = x_train.to(self.device)

            # split data read in train_loader
            in_new, ctr_new = x_train[:, :, :, :-1], x_train[:, 0, 0, -1]

            self.pred_model.eval()
            self.phys_model.eval()
            for param in list(self.pred_model.parameters()):
                param.requires_grad = False

            # 3 steps to generate new data along gradient
            for _ in range(self.params.phys_steps):
                ctr_new = ctr_new.requires_grad_(True)
                in_new = in_new.requires_grad_(True)
                pred, _, _, _ = self.pred_model(in_new, ctr_new)
                out_pred = pred[:, :, :, :3]
                mod = self.phys_model(in_new, ctr_new, out_pred)
                loss = ((Lpde(out_pred, in_new, self.dt) + mod) ** 2).mean()
                loss.backward()
                # print(ctr_new.is_leaf, in_new.is_leaf)
                dLf = ctr_new.grad
                dLu = in_new.grad
                # print(ctr_new.shape, in_new.shape)
                # print(dLu.shape, dLf.shape)
                phys_scale = self.params.phys_scale
                scale = torch.sqrt(loss.data) / ((dLf ** 2).sum() + (dLu ** 2).sum()) * phys_scale
                # print(f'scale:{scale1} {scale2}')
                ctr_new = ctr_new.data + scale * dLf    # use .data to generate new leaf tensor
                in_new = in_new.data + scale * dLu
                # print('f in : {:1.4e} {:1.4e}'.format((ctr_new ** 2).mean(), (in_new ** 2).mean()))
                # print('dLf dLu : {:1.4e} {:1.4e}'.format((dLf ** 2).mean(), (dLu ** 2).mean()))
                # print(ctr_new.mean(),in_new.mean())
            
            in_train, ctr_train = in_new.data, ctr_new.data
            
            for param in list(self.pred_model.parameters()):
                param.requires_grad = True
            
            self.pred_model.train()
            self.pred_optimizer.zero_grad()

            pred, _, _, _ = self.pred_model(in_train, ctr_train)
            out_pred = pred[:, :, :, :3]
            mod = self.phys_model(in_train, ctr_train, out_pred)
            loss = ((Lpde(out_pred, in_train, self.dt) + mod) ** 2).mean()
            loss.backward()
            self.pred_optimizer.step()
            loss_pde.update(loss.item(), self.params.batch_size)
        
        t4 = default_timer()
        print('----phys training: # {} {:1.2f} (pde): {:1.2e} | '.format(phys_epoch, t4-t3, loss_pde.avg))
    
    def load_state(self, pred_log, phys_log):
        self.pred_model.load_state_dict(pred_log)
        self.phys_model.load_state_dict(phys_log)

    def pred_loss(self, ipt, ctr, opt):
        opt, Cd, Cl = opt
        # put data into model
        pred, x_rec, ctr_rec, trans_out = self.pred_model(ipt, ctr)
        ipt_rec = x_rec[:, :, :, :3]
        # latent items
        opt_latent = self.pred_model.stat_en(opt)
        # prediction items
        opt_pred = pred[:, :, :, :3]
        Cd_pred = torch.mean(pred[:, :, :, -2].reshape(pred.shape[0], -1), 1)
        Cl_pred = torch.mean(pred[:, :, :, -1].reshape(pred.shape[0], -1), 1)
        loss1 = rel_error(opt_pred, opt).mean() + rel_error(Cd_pred, Cd).mean() + rel_error(Cl_pred, Cl).mean()
        loss2 = rel_error(ipt_rec, ipt).mean()
        loss3 = rel_error(ctr_rec, ctr).mean()
        loss4 = rel_error(trans_out, opt_latent).mean()
        return loss1, loss2, loss3, loss4

    def process(self, train_loader, test_loader, logs):
        for epoch in range(1, self.params.epochs+1):
            self.data_train_test(epoch, train_loader, test_loader, logs)
            if epoch % self.params.phys_gap == 0 and epoch != self.params.epochs:
                # freeze phys_model trained in data training
                for param in list(self.phys_model.parameters()):
                    param.requires_grad = False

                for phys_epoch in range(1, self.params.phys_epochs+1):
                    self.phys_train(phys_epoch, train_loader)
                
                for param in list(self.phys_model.parameters()):
                    param.requires_grad = True
            self.save_log(logs)

    def simulate(self, data_loader):
        self.pred_model.eval()
        self.phys_model.eval()

        loss1, loss2, loss3, loss4, loss5, loss6 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

        with torch.no_grad():
            for x_test, y_test in data_loader:
                x_test, y_test = x_test.to(self.device), y_test.to(self.device)
                # split data read in test_loader
                in_test, ctr_test = x_test[:, :, :, :-1], x_test[:, 0, 0, -1]
                out_test, Cd_test, Cl_test = y_test[:, :, :, :-2], y_test[:, 0, 0, -2], y_test[:, 0, 0, -1]
                opt_test = [out_test, Cd_test, Cl_test]
                loss_pred, loss_rec1, loss_rec2, loss_lat = self.pred_loss(in_test, ctr_test, opt_test)
                
                mod_test = self.phys_model(in_test, ctr_test, out_test)
                loss_pde_obs = ((Lpde(out_test, in_test, self.dt) + mod_test) ** 2).mean()

                pred, _, _, _ = self.pred_model(in_test, ctr_test)
                out_pred = pred[:, :, :, :3]
                mod_pred = self.phys_model(in_test, ctr_test, out_pred)
                loss_pde_pred = ((Lpde(out_pred, in_test, self.dt) + mod_pred) ** 2).mean()

                loss1.update(loss_pred.item(), self.params.batch_size)
                loss2.update(loss_rec1.item(), self.params.batch_size)
                loss3.update(loss_rec2.item(), self.params.batch_size)
                loss4.update(loss_lat.item(), self.params.batch_size)
                loss5.update(loss_pde_obs.item(), self.params.batch_size)
                loss6.update(loss_pde_pred.item(), self.params.batch_size)
                
        return loss1.avg, loss2.avg, loss3.avg, loss4.avg, loss5.avg, loss6.avg

    def save_log(self, logs):
        logs['pred_model'].append(self.pred_model.state_dict())
        logs['phys_model'].append(self.phys_model.state_dict())


class LoadModel():
    def __init__(self, operator_path, shape):
        # mosel params setting
        print(operator_path)
        state_dict_pred, state_dict_phys, logs_model = torch.load(operator_path)
        params_args = logs_model['args']
        L = params_args.L
        modes = params_args.modes
        width = params_args.width
        f_channels = params_args.f_channels

        self.data_norm = logs_model['data_norm']
        self.dt = params_args.tg* 0.01

        model_params = dict()
        model_params['modes'] = modes
        model_params['width'] = width
        model_params['L'] = L
        model_params['f_channels'] = f_channels
        model_params['shape'] = shape

        self.pred_model = FNO_ensemble(model_params)
        self.pred_model.load_state_dict(state_dict_pred)
        self.pred_model.eval()
        self.phys_model = state_mo(model_params)
        self.phys_model.load_state_dict(state_dict_phys)
        self.phys_model.eval()
    
    def cal_1step(self, obs, Cd, Cl, ctr):
        nt, nx, ny = obs.shape[0] - 1, obs.shape[1], obs.shape[2]
        out_nn = torch.zeros(nt, nx, ny, 3)
        Cd_nn, Cl_nn, Lpde_obs, Lpde_pred = torch.zeros(nt), torch.zeros(nt), torch.zeros(nt), torch.zeros(nt)
        for k in range(nt):
            pred, _, _, _ = self.pred_model(obs[k].unsqueeze(0), ctr[k].reshape(1))
            pred = pred[:, :, :, :3]
            out_nn[k] = pred.squeeze()
            Cd_nn[k] = torch.mean(pred[:, :, :, -2])
            Cl_nn[k] = torch.mean(pred[:, :, :, -1])
            mod_obs = self.phys_model(obs[k].unsqueeze(0), ctr[k].reshape(1), obs[k+1].unsqueeze(0))
            Lpde_obs[k] = ((Lpde(obs[k+1].unsqueeze(0), obs[k].unsqueeze(0), self.dt) + mod_obs) ** 2).mean()
            mod_pred = self.phys_model(obs[k].unsqueeze(0), ctr[k].reshape(1), pred)
            Lpde_pred[k] = ((Lpde(pred, obs[k].unsqueeze(0), self.dt) + mod_pred) ** 2).mean()
        
        Cd_mean, Cd_var = self.data_norm['Cd']
        Cl_mean, Cl_var = self.data_norm['Cl']
        Cd_nn = Cd_nn * Cd_var.item() + Cd_mean.item()
        Cl_nn = Cl_nn * Cl_var.item() + Cl_mean.item()
        error_1step = ((out_nn - obs[1:]) ** 2).reshape(nt, -1).mean(1) + ((Cd_nn - Cd) ** 2).reshape(nt, -1).mean(1) + ((Cl_nn - Cl) ** 2).reshape(nt, -1).mean(1)

        del out_nn, Cd_nn, Cl_nn
        gc.collect()
        return error_1step, Lpde_obs, Lpde_pred

    def process(self, obs, Cd, Cl, ctr, t_start):
        nt, nx, ny = obs.shape[0] - 1, obs.shape[1], obs.shape[2]
        out_nn = torch.zeros(nt, nx, ny, 3)
        Cd_nn, Cl_nn, Lpde_pred = torch.zeros(nt), torch.zeros(nt), torch.zeros(nt)
        for k in range(t_start, nt):
            pred, _, _, _ = self.pred_model(self.in_nn, ctr[k].reshape(1))
            pred = pred[..., :3]
            out_nn[k] = pred.squeeze()
            mod_pred = self.phys_model(self.in_nn, ctr[k].reshape(1), pred)
            self.in_nn = pred
            Lpde_pred[k] = ((Lpde(pred, self.in_nn, self.dt) + mod_pred) ** 2).mean()
            Cd_nn[k] = torch.mean(pred[:, :, :, -2])
            Cl_nn[k] = torch.mean(pred[:, :, :, -1])

        Cd_mean, Cd_var = self.data_norm['Cd']
        Cl_mean, Cl_var = self.data_norm['Cl']
        Cd_nn = Cd_nn * Cd_var.item() + Cd_mean.item()
        Cl_nn = Cl_nn * Cl_var.item() + Cl_mean.item()

        out_nn[:t_start], Cd_nn[:t_start], Cl_nn[:t_start] = obs[1:1+t_start], Cd[:t_start], Cl[:t_start]
        error_cul = ((out_nn - obs[1:]) ** 2).reshape(nt, -1).mean(1) + ((Cd_nn - Cd) ** 2).reshape(nt, -1).mean(1) + ((Cl_nn - Cl) ** 2).reshape(nt, -1).mean(1)

        del out_nn, Cd_nn, Cl_nn
        gc.collect()
        return error_cul, Lpde_pred
    
    def set_init(self, state_nn):
        self.in_nn = state_nn

    