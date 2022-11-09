from email.policy import default
import torch
from torch.utils.data import DataLoader
from timeit import default_timer
import copy

from scripts_test.nets import *
from scripts.utils import *

class NSEModel():
    def __init__(self, shape, dt, args):
        self.shape = shape
        self.dt = dt
        self.params = args
        self.Re = 0.001
        self.Lx = 2.2
        self.Ly = 0.41
        self.device = torch.device('cuda:{}'.format(self.params.gpu) if torch.cuda.is_available() else 'cpu')

    def set_model(self, pred_model=FNO_ensemble, phys_model=state_mo):

        model_params = dict()
        model_params['modes'] = self.params.modes
        model_params['width'] = self.params.width
        model_params['L'] = self.params.L
        model_params['shape'] = self.shape
        model_params['f_channels'] = self.params.f_channels
        model_params['Lxy'] = [self.Lx, self.Ly]

        self.pred_model = pred_model(model_params).to(self.device)
        self.pred_optimizer = torch.optim.Adam(self.pred_model.parameters(), lr=self.params.lr, weight_decay=self.params.wd)
        self.pred_scheduler = torch.optim.lr_scheduler.StepLR(self.pred_optimizer, step_size=self.params.step_size, gamma=self.params.gamma)

        self.phys_model = phys_model(model_params).to(self.device)
        self.phys_optimizer = torch.optim.Adam(self.phys_model.parameters(), lr=self.params.lr * 5, weight_decay=self.params.wd)
        self.phys_scheduler = torch.optim.lr_scheduler.StepLR(self.phys_optimizer, step_size=self.params.step_size, gamma=self.params.gamma)

    def toCPU(self):
        self.pred_model.to('cpu')
        self.phys_model.to('cpu')

    def count_params(self):
        c = 0
        for p in list(self.pred_model.parameters()):
            c += reduce(operator.mul, list(p.size()))
        for p in list(self.phys_model.parameters()):
            c += reduce(operator.mul, list(p.size()))
        return c

    def load_state(self, pred_log, phys_log):
        self.pred_model.load_state_dict(pred_log)
        self.phys_model.load_state_dict(phys_log)
        self.pred_model.eval()
        self.phys_model.eval()
    
    def save_log(self, logs):
        with torch.no_grad():
            logs['pred_model'].append(copy.deepcopy(self.pred_model.state_dict()))
            logs['phys_model'].append(copy.deepcopy(self.phys_model.state_dict()))
    
    def set_init(self, state_nn):
        self.in_nn = state_nn

    def data_train(self, epoch, train_loader):
        self.pred_model.train()
        self.phys_model.train()

        t1 = default_timer()
        train_log = PredLog(length=self.params.batch_size)
        
        for x_train, y_train in train_loader:
            x_train, y_train = x_train.to(self.device), y_train.to(self.device)
            
            self.pred_optimizer.zero_grad()
            self.phys_optimizer.zero_grad()
            
            # split data read in train_loader
            in_train, ctr_train = x_train[:, :, :, :-1], x_train[:, 0, 0, -1]
            out_train = y_train[:, :, :, :3]
            opt_train = y_train

            # put data to generate 4 loss
            loss1, loss2, loss3, loss4, loss6 = self.pred_loss(in_train, ctr_train, opt_train)
            mod = self.phys_model(in_train, ctr_train)
            loss5 = ((Lpde(in_train, out_train, self.dt, Re = self.Re, Lx = self.Lx, Ly = self.Ly) + mod) ** 2).mean()

            self.train_step(loss1, loss2, loss3, loss4, loss5, loss6)

            train_log.update([loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), loss6.item()])
        
        self.scheduler_step()
        t2 = default_timer()

        print('# {} train: {:1.2f} | (pred): {:1.2e}  (rec) state: {:1.2e}  ctr: {:1.2e} (latent): {:1.2e} (pde) obs: {:1.2e} pred: {:1.2e}'
              .format(epoch, t2-t1, train_log.loss1.avg, train_log.loss2.avg, train_log.loss3.avg, train_log.loss4.avg, train_log.loss5.avg, train_log.loss6.avg))

    def test(self, test_loader, logs):
        self.pred_model.eval()
        self.phys_model.eval()
        test_log = PredLog(length=self.params.batch_size)
        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test, y_test = x_test.to(self.device), y_test.to(self.device)
                # split data read in test_loader
                in_test, ctr_test = x_test[:, :, :, :-1], x_test[:, 0, 0, -1]
                out_test = y_test[:, :, :, :3]
                opt_test = y_test

                loss1, loss2, loss3, loss4, loss6 = self.pred_loss(in_test, ctr_test, opt_test)
                mod = self.phys_model(in_test, ctr_test)
                loss5 = ((Lpde(in_test, out_test, self.dt, Re = self.Re, Lx = self.Lx, Ly = self.Ly) + mod) ** 2).mean()
                test_log.update([loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), loss6.item()])
            test_log.save_log(logs)
        
        print('--test | (pred): {:1.2e}  (rec) state: {:1.2e}  ctr: {:1.2e} (latent): {:1.2e} (pde) obs: {:1.2e} pred: {:1.2e}'
              .format(test_log.loss1.avg, test_log.loss2.avg, test_log.loss3.avg, test_log.loss4.avg, test_log.loss5.avg, test_log.loss6.avg))

    def cal_1step(self, data):
        obs, Cd, Cl, ctr = data.get_data()
        obs_ = obs + (2 * torch.rand(obs.shape) - 1) * 0.00
        ctr_ = ctr + (2 * torch.rand(ctr.shape) - 1) * 0.00
        N0, nt = obs.shape[0], obs.shape[1] - 1
        nx, ny = self.shape
        out_nn, Lpde_obs, Lpde_pred = torch.zeros(N0, nt, nx, ny, 3), torch.zeros(N0, nt, nx, ny, 2), torch.zeros(N0, nt, nx, ny, 2)
        Cd_nn, Cl_nn = torch.zeros(N0, nt), torch.zeros(N0, nt)
        error_1step, error_Cd, error_Cl = torch.zeros(N0, nt), torch.zeros(N0, nt), torch.zeros(N0, nt)
        with torch.no_grad():
            for k in range(nt):
                t1 = default_timer()
                out_nn[:, k], Cd_nn[:, k], Cl_nn[:, k], mod_pred, _, _, _ = self.model_step(obs_[:, k], ctr_[:, k])
                Lpde_pred[:, k] = ((Lpde(obs_[:, k], out_nn[:, k], self.dt, self.Re, Lx = self.Lx, Ly = self.Ly) + mod_pred) ** 2)

                mod_obs = self.phys_model(obs[:, k], ctr[:, k])
                Lpde_obs[:, k] = ((Lpde(obs[:, k], obs[:, k+1], self.dt, self.Re, Lx = self.Lx, Ly = self.Ly) + mod_obs) ** 2)
                
                error_1step[:, k] = rel_error(out_nn[:, k], obs[:, k+1]) 
                error_Cd[:, k] = ((Cd_nn[:, k] - Cd[:, k]) ** 2)
                error_Cl[:, k] = ((Cl_nn[:, k] - Cl[:, k]) ** 2)
                t2 = default_timer()
                if k % 10 == 0:
                    print(f'# {k} | {t2 - t1:1.2f}: error_Cd: {error_Cd[:, k].mean():1.4f} | error_Cl: {error_Cl[:, k].mean():1.4f} | error_state: {error_1step[:, k].max():1.4f}\
                        | pred_Lpde: {Lpde_pred[:, k].mean():1.4f} | obs_Lpde: {Lpde_obs[:, k].mean():1.4f}')

        return error_1step, Lpde_obs, Lpde_pred

    def process(self, data):
        obs, Cd, Cl, ctr = data.get_data()
        self.set_init(obs[:, 0])
        N0, nt = obs.shape[0], obs.shape[1] - 1
        nx, ny = self.shape
        print(f'N0: {N0}, nt: {nt}, nx: {nx}, ny: {ny}')
        out_nn, Lpde_pred = torch.zeros(N0, nt, nx, ny, 3), torch.zeros(N0, nt, nx, ny, 2)
        Cd_nn, Cl_nn = torch.zeros(N0, nt), torch.zeros(N0, nt)
        error_cul, error_Cd, error_Cl = torch.zeros(N0, nt), torch.zeros(N0, nt), torch.zeros(N0, nt)
        with torch.no_grad():
            for k in range(nt):
                t1 = default_timer()
                out_nn[:, k], Cd_nn[:, k], Cl_nn[:, k], mod_pred, _, _, _ = self.model_step(self.in_nn, ctr[:, k])
                # print(pred.shape, mod_pred.shape, self.in_nn.shape)
                Lpde_pred[:, k] = ((Lpde(self.in_nn, out_nn[:, k], self.dt, self.Re) + mod_pred) ** 2)
                self.in_nn = out_nn[:, k]
                error_cul[:, k] = rel_error(out_nn[:, k], obs[:, k+1]) 
                error_Cd[:, k] = ((Cd_nn[:, k] - Cd[:, k]) ** 2)
                error_Cl[:, k] = ((Cl_nn[:, k] - Cl[:, k]) ** 2)
                t2 = default_timer()
                if k % 10 == 0:
                    print(f'# {k} | {t2 - t1:1.2f}: error_Cd: {error_Cd[:, k].mean():1.4f} | error_Cl: {error_Cl[:, k].mean():1.4f} | \
                            error_state: {error_cul[:, k].mean():1.4f}| cul_Lpde: {Lpde_pred[:, k].mean():1.4f}')

        return out_nn, Lpde_pred

    def pred_loss(self, ipt, ctr, opt):
        out, Cd, Cl = opt[:, :, :, :3], opt[:, 0, 0, -2], opt[:, 0, 0, -1]
        # latent items
        out_latent = self.pred_model.stat_en(out)
        # prediction & rec items
        out_pred, Cd_pred, Cl_pred, mod_pred, ipt_rec, ctr_rec, trans_out = self.model_step(ipt, ctr)
        
        loss1 = rel_error(out_pred, out).mean() + rel_error(Cd_pred, Cd).mean() + rel_error(Cl_pred, Cl).mean()
        loss2 = rel_error(ipt_rec, ipt).mean()
        loss3 = rel_error(ctr_rec, ctr).mean()
        loss4 = rel_error(trans_out, out_latent).mean()
        loss6 = ((Lpde(ipt, out_pred, self.dt, Re = self.Re) + mod_pred) ** 2).mean()

        return loss1, loss2, loss3, loss4, loss6

    def model_step(self, ipt, ctr):
        pred, x_rec, ctr_rec, trans_out = self.pred_model(ipt, ctr)
        ipt_rec = x_rec[:, :, :, :3]
        out_pred = pred[:, :, :, :3]
        Cd_pred = torch.mean(pred[:, :, :, -2].reshape(pred.shape[0], -1), 1)
        Cl_pred = torch.mean(pred[:, :, :, -1].reshape(pred.shape[0], -1), 1)
        mod_pred = self.phys_model(ipt, ctr)
        return out_pred, Cd_pred, Cl_pred, mod_pred, ipt_rec, ctr_rec, trans_out

    def train_step(self, loss1, loss2, loss3, loss4, loss5, loss6):
        print('to be finished')
        # do backward(), optim.step()
        pass
    
    def scheduler_step(self):
        print('to be finished')
        pass
        

class NSEModel_FNO(NSEModel):
    def __init__(self, shape, dt, args):
        super().__init__(shape, dt, args)
        self.set_model()
    
    def train_step(self, loss1, loss2, loss3, loss4, loss5, loss6):
        lambda1, lambda2, lambda3, lambda4 = self.params.lambda1, self.params.lambda2, self.params.lambda3, self.params.lambda4
        loss_pred = lambda1 * loss1 + lambda2 * loss2 + lambda3 * loss3 + lambda4 * loss4

        loss_pred.backward()
        loss5.backward()

        self.pred_optimizer.step()
        self.phys_optimizer.step()
    
    def scheduler_step(self):
        self.pred_scheduler.step()
        self.phys_scheduler.step()

    def phys_train(self, phys_epoch, train_loader, random=False):
        loss_pde = AverageMeter()
        t3 = default_timer()

        for x_train, _ in train_loader:
            x_train = x_train.to(self.device)

            # split data read in train_loader
            in_new, ctr_new = x_train[:, :, :, :-1], x_train[:, 0, 0, -1]

            self.phys_model.eval()

            in_train, ctr_train = self.gen_new_data(in_new, ctr_new, random)
            
            self.pred_model.train()
            self.pred_optimizer.zero_grad()

            pred, _, _, _ = self.pred_model(in_train, ctr_train)
            out_pred = pred[:, :, :, :3]
            mod = self.phys_model(in_train, ctr_train)
            # 多训练几次？  
            loss = ((Lpde(in_train, out_pred, self.dt) + mod) ** 2).mean()
            loss.backward()
            self.pred_optimizer.step()
            loss_pde.update(loss.item(), self.params.batch_size)
        
        self.phys_scheduler.step()
        t4 = default_timer()
        print('----phys training: # {} {:1.2f} (pde) pred: {:1.2e} | '.format(phys_epoch, t4-t3, loss_pde.avg))
    
    def gen_new_data(self, in_new, ctr_new, random=False):
        if random == True:
            in_train = in_new + torch.rand(in_new.shape).cuda() * self.params.phys_scale
            ctr_train = ctr_new + torch.rand(ctr_new.shape).cuda() * self.params.phys_scale
            return in_train, ctr_train

        self.pred_model.eval()
        for param in list(self.pred_model.parameters()):
            param.requires_grad = False

        # 3 steps to generate new data along gradient
        if self.params.phys_scale > 0:
            for _ in range(self.params.phys_steps):
                ctr_new = ctr_new.requires_grad_(True)
                in_new = in_new.requires_grad_(True)
                pred, _, _, _ = self.pred_model(in_new, ctr_new)
                out_pred = pred[:, :, :, :3]
                mod = self.phys_model(in_new, ctr_new)
                loss = ((Lpde(in_new, out_pred, self.dt) + mod) ** 2).mean()
                loss.backward()
                # print(ctr_new.is_leaf, in_new.is_leaf)
                dLf = ctr_new.grad
                dLu = in_new.grad
                # print(ctr_new.shape, in_new.shape)
                # print(dLu.shape, dLf.shape)
                phys_scale = self.params.phys_scale
                scale = torch.sqrt(loss.data) / torch.sqrt((dLf ** 2).sum() + (dLu ** 2).sum()) * phys_scale
                ctr_new = ctr_new.data + scale * dLf    # use .data to generate new leaf tensor
                in_new = in_new.data + scale * dLu
        
        in_train, ctr_train = in_new.data, ctr_new.data
        
        for param in list(self.pred_model.parameters()):
            param.requires_grad = True

        return in_train, ctr_train
        
        
class NSEModel_FNO_test(NSEModel):
    def __init__(self, shape, dt, args):
        super().__init__(shape, dt, args)
        self.set_model()
    
    def train_step(self, loss1, loss2, loss3, loss4, loss5, loss6):
        lambda1, lambda2, lambda3, lambda4 = self.params.lambda1, self.params.lambda2, self.params.lambda3, self.params.lambda4
        loss_pred = lambda1 * loss1 + lambda2 * loss2 + lambda3 * loss3 + lambda4 * loss4 + 0.1 * loss6

        loss_pred.backward()
        self.pred_optimizer.step()
    
    def scheduler_step(self):
        self.pred_scheduler.step()
    
    def pred_loss(self, ipt, ctr, opt):
        out, Cd, Cl = opt[:, :, :, :3], opt[:, 0, 0, -2], opt[:, 0, 0, -1]
        # latent items
        out_latent = self.pred_model.stat_en(out)
        # prediction & rec items
        out_pred, Cd_pred, Cl_pred, mod_pred, ipt_rec, ctr_rec, trans_out = self.model_step(ipt, ctr)
        
        loss1 = rel_error(out_pred, out).mean() + rel_error(Cd_pred, Cd).mean() + rel_error(Cl_pred, Cl).mean()
        loss2 = rel_error(ipt_rec, ipt).mean()
        loss3 = rel_error(ctr_rec, ctr).mean()
        loss4 = rel_error(trans_out, out_latent).mean()
        loss6 = ((Lpde(ipt, out_pred, self.dt, Re = self.Re)) ** 2).mean()

        return loss1, loss2, loss3, loss4, loss6

