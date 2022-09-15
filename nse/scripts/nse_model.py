import torch
from torch.utils.data import DataLoader
from timeit import default_timer

from scripts.models import *
from scripts.utils import *

class NSEModel_FNO:
    def __init__(self, args, shape, dt, logs, modify):
        self.logs = logs
        self.params = args
        self.modify = modify
        self.dt = dt
        self.device = torch.device('cuda:{}'.format(self.params.gpu) if torch.cuda.is_available() else 'cpu')

        model_params = dict()
        model_params['modes'] = self.params.modes
        model_params['width'] = self.params.width
        model_params['L'] = self.params.L
        model_params['shape'] = shape
        model_params['f_channels'] = self.params.f_channels

        self.epochs = self.params.epochs
        self.batch_size = self.params.batch_size
        lr = self.params.lr
        wd = self.params.wd
        step_size = self.params.step_size
        gamma = self.params.gamma        
        
        self.model = FNO_ensemble(model_params).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

    def count_params(self):
        c = 0
        for p in list(self.model.parameters()):
            c += reduce(operator.mul, list(p.size()))
        return c
    
    def train_test(self, epoch, train_loader, test_loader):
        lambda1, lambda2, lambda3, lambda4, lambda5 = self.params.lambda1, self.params.lambda2, \
                                                      self.params.lambda3, self.params.lambda4, self.params.lambda5
        self.model.train()

        t1 = default_timer()
        train_loss = AverageMeter()
        train_loss1 = AverageMeter()
        train_loss2 = AverageMeter()
        train_loss3 = AverageMeter()
        train_loss4 = AverageMeter()
        train_loss5 = AverageMeter()

        device = self.device
        for x_train, y_train in train_loader:
            x_train, y_train = x_train.to(device), y_train.to(device)
            
            self.optimizer.zero_grad()

            # split data read in train_loader
            in_train, ctr_train = x_train[:, :, :, :-1], x_train[:, 0, 0, -1]
            out_train, Cd_train, Cl_train = y_train[:, :, :, :-2], y_train[:, 0, 0, -2], y_train[:, 0, 0, -1]
            opt_train = [out_train, Cd_train, Cl_train]

            # put data to generate 4 loss
            opt_pred, loss1, loss2, loss3, loss4 = self.pred_step(in_train, ctr_train, opt_train)
            # physical loss
            mod = self.model.state_mo(in_train, ctr_train, out_train)
            loss_pde = ((Lpde(opt_pred, in_train, self.dt) + mod) ** 2).mean()
    
            loss = lambda1 * loss1 + lambda2 * loss2 + lambda3 * loss3 + lambda4 * loss4 + lambda5 * loss_pde
            
            loss.backward()
            self.optimizer.step()

            train_loss.update(loss.item(), x_train.shape[0])
            train_loss1.update(loss1.item(), x_train.shape[0])
            train_loss2.update(loss2.item(), x_train.shape[0])
            train_loss3.update(loss3.item(), x_train.shape[0])
            train_loss4.update(loss4.item(), x_train.shape[0])
            train_loss5.update(loss_pde.item(), x_train.shape[0])
        
        if epoch % 1 == 0:
            for x_train, _ in train_loader:
                x_train = x_train.to(device)

                # split data read in train_loader
                in_new, f_new = x_train[:, :, :, :-1], x_train[:, 0, 0, -1]

                for param in list(self.model.parameters()):
                    param.requires_grad = False

                # 3 steps to generate new data along gradient
                for i in range(3):
                    f_new = f_new.requires_grad_()
                    in_new = in_new.requires_grad_()
                    pred, _, _, _ = self.model(in_new, f_new)
                    out_pred = pred[:, :, :, :3]
                    mod = self.model.state_mo(in_new, f_new, out_pred)
                    loss_pde = ((Lpde(out_pred, in_new, self.dt) + mod) ** 2).sum()
                    loss_pde.backward()
                    print(f_new.is_leaf, in_new.is_leaf)
                    # dLf = f_new.grad
                    dLu = in_new.grad
                    
                    # print(f_new.shape, in_new.shape)
                    print(dLu.grad)
                    # print(dLf.shape)
                    scale = 0.01
                    # f_new = f_new + scale * dLf
                    in_new = in_new + scale * dLu
                
                for param in list(self.model.parameters()):
                    param.requires_grad = True
                for param in list(self.model.state_mo.parameters()):
                    param.requires_grad = False
                
                self.model.train()
                self.optimizer.zero_grad()

                in_train, f_train = in_new.data, f_new.data

                pred, _, _, _ = self.model(in_train, f_train)
                out_pred = pred[:, :, :, :3]
                mod = self.model.state_mo(in_train, f_train, out_pred)
                loss_pde = ((Lpde(out_pred, in_train, self.dt) + mod) ** 2).mean()
                loss_pde.backward()
                self.optimizer.step()

        
        self.logs['train_loss'].append(train_loss.avg)
        self.logs['train_loss_f_t_rec'].append(train_loss3.avg)
        self.logs['train_loss_u_t_rec'].append(train_loss2.avg)
        self.logs['train_loss_trans'].append(train_loss1.avg)
        self.logs['train_loss_trans_latent'].append(train_loss4.avg)
        self.logs['train_loss_pde'].append(train_loss5.avg)
        
        self.scheduler.step()
        t2 = default_timer()
        
        self.model.eval()

        test_loss = AverageMeter()
        test_loss1 = AverageMeter()
        test_loss2 = AverageMeter()
        test_loss3 = AverageMeter()
        test_loss4 = AverageMeter()
        test_loss5 = AverageMeter()

        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test, y_test = x_test.to(device), y_test.to(device)

                # split data read in test_loader
                in_test, ctr_test = x_test[:, :, :, :-1], x_test[:, 0, 0, -1]
                out_test, Cd_test, Cl_test = y_test[:, :, :, :-2], y_test[:, 0, 0, -2], y_test[:, 0, 0, -1]
                opt_test = [out_test, Cd_test, Cl_test]
                opt_pred, loss1, loss2, loss3, loss4 = self.pred_step(in_test, ctr_test, opt_test)
                mod = self.model.state_mo(in_test, ctr_test, out_test)
                loss_pde = ((Lpde(opt_pred, in_test, self.dt) + mod) ** 2).mean()

                loss = lambda1 * loss1 + lambda2 * loss2 + lambda3 * loss3 + lambda4 * loss4 + lambda5 * loss_pde

                test_loss.update(loss.item(), x_test.shape[0])
                test_loss1.update(loss1.item(), x_test.shape[0])
                test_loss2.update(loss2.item(), x_test.shape[0])
                test_loss3.update(loss3.item(), x_test.shape[0])
                test_loss4.update(loss4.item(), x_test.shape[0])
                test_loss5.update(loss_pde.item(), x_test.shape[0])
            
            self.logs['test_loss'].append(test_loss.avg)
            self.logs['test_loss_f_t_rec'].append(test_loss3.avg)
            self.logs['test_loss_u_t_rec'].append(test_loss2.avg)
            self.logs['test_loss_trans'].append(test_loss1.avg)
            self.logs['test_loss_trans_latent'].append(test_loss4.avg)
            self.logs['test_loss_pde'].append(test_loss5.avg)
        
        print('# {} {:1.3f} | loss1: {:1.2e}  loss2: {:1.2e}  loss3: {:1.2e} loss4: {:1.2e} loss5: {:1.2e} | loss1: {:1.2e} loss2: {:1.2e}  loss3: {:1.2e} loss4: {:1.2e} loss5: {:1.2e}'
              .format(epoch, t2-t1, train_loss1.avg, train_loss2.avg, train_loss3.avg, train_loss4.avg, train_loss5.avg, test_loss1.avg, test_loss2.avg, test_loss3.avg, test_loss4.avg, test_loss5.avg))

    def pred_step(self, ipt, ctr, opt):
        opt, Cd, Cl = opt

        # put data into model
        pred, x_rec, ctr_rec, trans_out = self.model(ipt, ctr)
        ipt_rec = x_rec[:, :, :, :3]

        # latent items
        opt_latent = self.model.stat_en(opt)

        # prediction items
        opt_pred = pred[:, :, :, :3]
        Cd_pred = torch.mean(pred[:, :, :, -2].reshape(self.batch_size, -1), 1)
        Cl_pred = torch.mean(pred[:, :, :, -1].reshape(self.batch_size, -1), 1)

        loss1 = rel_error(opt_pred, opt).mean()\
                + rel_error(Cd_pred, Cd).mean()\
                + rel_error(Cl_pred, Cl).mean()
        loss2 = rel_error(ipt_rec, ipt).mean()
        loss3 = rel_error(ctr_rec, ctr).mean()
        loss4 = rel_error(trans_out, opt_latent).mean()

        return opt_pred, loss1, loss2, loss3, loss4

    def process(self, train_loader, test_loader):
        for epoch in range(1, self.epochs+1):
            self.train_test(epoch, train_loader, test_loader)


class NSEModel_PIPN:
    def __init__(self, args, dt, logs):
        self.logs = logs
        self.params = args
        self.device = torch.device('cuda:{}'.format(self.params.gpu) if torch.cuda.is_available() else 'cpu')

        self.epochs = self.params.epochs
        self.batch_size = self.params.batch_size
        lr = self.params.lr
        wd = self.params.wd
        step_size = self.params.step_size
        gamma = self.params.gamma

        self.dt = dt
        
        self.model = PIPN().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

    def count_params(self):
        c = 0
        for p in list(self.model.parameters()):
            c += reduce(operator.mul, list(p.size()))
        return c
    
    def train_test(self, epoch, train_loader, test_loader):
        lambda1, lambda5 = self.params.lambda1, self.params.lambda5
        self.model.train()

        t1 = default_timer()
        train_loss = AverageMeter()
        train_loss1 = AverageMeter()
        train_loss5 = AverageMeter()

        device = self.device
        for x_train, y_train in train_loader:
            x_train, y_train = x_train.to(device), y_train.to(device)
            
            self.optimizer.zero_grad()

            # split data read in train_loader
            in_train, f_train = x_train[..., :-1], x_train[:, 0, -1]
            out_train, Cd_train, Cl_train = y_train[:, :, 2:-2], y_train[:, 0, -2], y_train[:, 0, -1]
            # put data into model
            pred = self.model(in_train, f_train)
            # prediction items
            out_pred = pred[..., :3]
            Cd_pred = torch.mean(pred[..., -2].reshape(self.batch_size, -1), 1)
            Cl_pred = torch.mean(pred[..., -1].reshape(self.batch_size, -1), 1)

            # loss1: prediction loss; loss2: rec loss of state
            # loss3: rec loss of f; loss4: latent loss
            loss1 = rel_error(out_pred, out_train).mean()\
                    + rel_error(Cd_pred, Cd_train).mean()\
                    + rel_error(Cl_pred, Cl_train).mean()
            # loss_pde = Lpde(out_pred, in_train, self.dt)
            loss = lambda1 * loss1 # + lambda5 * loss_pde
            
            loss.backward()
            self.optimizer.step()

            train_loss.update(loss.item(), x_train.shape[0])
            train_loss1.update(loss1.item(), x_train.shape[0])
            # train_loss5.update(loss_pde.item(), x_train.shape[0])
        
        self.logs['train_loss'].append(train_loss.avg)
        self.logs['train_loss_trans'].append(train_loss1.avg)
        self.logs['train_loss_pde'].append(train_loss5.avg)
        
        self.scheduler.step()
        t2 = default_timer()
        
        self.model.eval()

        test_loss = AverageMeter()
        test_loss1 = AverageMeter()
        test_loss5 = AverageMeter()

        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test, y_test = x_test.to(device), y_test.to(device)

                # split data read in test_loader
                in_test, f_test = x_test[..., :-1], x_test[:, 0, -1]
                out_test, Cd_test, Cl_test = y_test[:, :, 2:-2], y_test[:, 0, -2], y_test[:, 0, -1]
                # put data into model
                pred = self.model(in_test, f_test)
                # prediction items
                out_pred = pred[..., :3]
                Cd_pred = torch.mean(pred[..., -2].reshape(self.batch_size, -1), 1)
                Cl_pred = torch.mean(pred[..., -1].reshape(self.batch_size, -1), 1)
                loss1 = rel_error(out_pred, out_test).mean()\
                        + rel_error(Cd_pred, Cd_test).mean()\
                        + rel_error(Cl_pred, Cl_test).mean()
                # loss_pde = Lpde(out_pred, in_test, self.dt)
                loss = lambda1 * loss1  # + lambda5 * loss_pde

                test_loss.update(loss.item(), x_test.shape[0])
                test_loss1.update(loss1.item(), x_test.shape[0])
                # test_loss5.update(loss_pde.item(), x_test.shape[0])
            
            self.logs['test_loss'].append(test_loss.avg)
            self.logs['test_loss_trans'].append(test_loss1.avg)
            self.logs['test_loss_pde'].append(test_loss5.avg)
        
        print('# {} {:1.3f} | loss1: {:1.2e}  loss5: {:1.2e} | loss1: {:1.2e} loss5: {:1.2e}'
              .format(epoch, t2-t1, train_loss1.avg, train_loss5.avg, test_loss1.avg, test_loss5.avg))

    def process(self, train_loader, test_loader):
        for epoch in range(1, self.epochs+1):
            self.train_test(epoch, train_loader, test_loader)


class NSEModel_FNO_test:
    def __init__(self, args, shape, dt, logs, modify):
        self.logs = logs
        self.params = args
        self.modify = modify
        self.device = torch.device('cuda:{}'.format(self.params.gpu) if torch.cuda.is_available() else 'cpu')

        model_params = dict()
        model_params['modes'] = self.params.modes
        model_params['width'] = self.params.width
        model_params['L'] = self.params.L
        model_params['shape'] = shape
        model_params['f_channels'] = self.params.f_channels

        self.epochs = self.params.epochs
        self.batch_size = self.params.batch_size
        lr = self.params.lr
        wd = self.params.wd
        step_size = self.params.step_size
        gamma = self.params.gamma

        self.dt = dt
        
        self.model = FNO_ensemble_test(model_params).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

    def count_params(self):
        c = 0
        for p in list(self.model.parameters()):
            c += reduce(operator.mul, list(p.size()))
        return c
    
    def train_test(self, epoch, train_loader, test_loader):
        lambda1, lambda2, lambda3, lambda4, lambda5 = self.params.lambda1, self.params.lambda2, \
                                                      self.params.lambda3, self.params.lambda4, self.params.lambda5
        self.model.train()

        t1 = default_timer()
        train_loss = AverageMeter()
        train_loss1 = AverageMeter()
        train_loss2 = AverageMeter()
        train_loss3 = AverageMeter()
        train_loss4 = AverageMeter()
        train_loss5 = AverageMeter()

        device = self.device
        for x_train, y_train in train_loader:
            x_train, y_train = x_train.to(device), y_train.to(device)
            
            self.optimizer.zero_grad()

            # split data read in train_loader
            in_train, f_train = x_train[:, :, :, :-1], x_train[:, 0, 0, -1]
            out_train, Cd_train, Cl_train = y_train[:, :, :, :-2], y_train[:, 0, 0, -2], y_train[:, 0, 0, -1]
            # put data into model
            pred, x_rec, f_rec, trans_out = self.model(in_train, f_train, self.modify)
            out_latent = self.model.stat_en(out_train)
            in_rec = x_rec[:, :, :, :3]
            # prediction items
            out_pred = pred[:, :, :, :3]
            in_mod = self.model.state_mo(in_train, self.modify)
            out_mod = self.model.state_mo(out_pred, self.modify)

            Cd_pred = torch.mean(pred[:, :, :, -2].reshape(self.batch_size, -1), 1)
            Cl_pred = torch.mean(pred[:, :, :, -1].reshape(self.batch_size, -1), 1)

            # loss1: prediction loss; loss2: rec loss of state
            # loss3: rec loss of f; loss4: latent loss
            loss1 = rel_error(out_pred, out_train).mean()\
                    + rel_error(Cd_pred, Cd_train).mean()\
                    + rel_error(Cl_pred, Cl_train).mean()
            loss2 = rel_error(in_rec, in_train + in_mod).mean()
            loss3 = rel_error(f_rec, f_train).mean()
            # loss3 = F.mse_loss(f_rec, f_train, reduction='mean')
            loss4 = rel_error(trans_out, out_latent).mean()
            loss_pde = (Lpde(out_pred + out_mod, in_train + in_mod, self.dt) ** 2).mean()
            loss = lambda1 * loss1 + lambda2 * loss2 + lambda3 * loss3 + lambda4 * loss4 + lambda5 * loss_pde
            
            loss.backward()
            self.optimizer.step()

            train_loss.update(loss.item(), x_train.shape[0])
            train_loss1.update(loss1.item(), x_train.shape[0])
            train_loss2.update(loss2.item(), x_train.shape[0])
            train_loss3.update(loss3.item(), x_train.shape[0])
            train_loss4.update(loss4.item(), x_train.shape[0])
            train_loss5.update(loss_pde.item(), x_train.shape[0])
        
        self.logs['train_loss'].append(train_loss.avg)
        self.logs['train_loss_f_t_rec'].append(train_loss3.avg)
        self.logs['train_loss_u_t_rec'].append(train_loss2.avg)
        self.logs['train_loss_trans'].append(train_loss1.avg)
        self.logs['train_loss_trans_latent'].append(train_loss4.avg)
        self.logs['train_loss_pde'].append(train_loss5.avg)
        
        self.scheduler.step()
        t2 = default_timer()
        
        self.model.eval()

        test_loss = AverageMeter()
        test_loss1 = AverageMeter()
        test_loss2 = AverageMeter()
        test_loss3 = AverageMeter()
        test_loss4 = AverageMeter()
        test_loss5 = AverageMeter()

        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test, y_test = x_test.to(device), y_test.to(device)

                # split data read in test_loader
                in_test, f_test = x_test[:, :, :, :-1], x_test[:, 0, 0, -1]
                out_test, Cd_test, Cl_test = y_test[:, :, :, :-2], y_test[:, 0, 0, -2], y_test[:, 0, 0, -1]
                # put data into model
                pred, x_rec, f_rec, trans_out = self.model(in_test, f_test, self.modify)
                out_latent = self.model.stat_en(out_test)
                in_rec = x_rec[:, :, :, :3]
                # prediction items
                out_pred = pred[:, :, :, :3]
                in_mod = self.model.state_mo(in_test, self.modify)
                out_mod = self.model.state_mo(out_pred, self.modify)

                Cd_pred = torch.mean(pred[:, :, :, -2].reshape(self.batch_size, -1), 1)
                Cl_pred = torch.mean(pred[:, :, :, -1].reshape(self.batch_size, -1), 1)

                loss1 = rel_error(out_pred, out_test).mean()\
                        + rel_error(Cd_pred, Cd_test).mean()\
                        + rel_error(Cl_pred, Cl_test).mean()
                loss2 = rel_error(in_rec, in_test + in_mod).mean()
                loss3 = rel_error(f_rec, f_test).mean()
                loss4 = rel_error(trans_out, out_latent).mean()
                loss_pde = (Lpde(out_pred + out_mod, in_test + in_mod, self.dt) ** 2).mean()
                # loss_pde = self.model.state_mo(in_test, f_test, self.modify)
                loss = lambda1 * loss1 + lambda2 * loss2 + lambda3 * loss3 + lambda4 * loss4 + lambda5 * loss_pde

                test_loss.update(loss.item(), x_test.shape[0])
                test_loss1.update(loss1.item(), x_test.shape[0])
                test_loss2.update(loss2.item(), x_test.shape[0])
                test_loss3.update(loss3.item(), x_test.shape[0])
                test_loss4.update(loss4.item(), x_test.shape[0])
                test_loss5.update(loss_pde.item(), x_test.shape[0])
            
            self.logs['test_loss'].append(test_loss.avg)
            self.logs['test_loss_f_t_rec'].append(test_loss3.avg)
            self.logs['test_loss_u_t_rec'].append(test_loss2.avg)
            self.logs['test_loss_trans'].append(test_loss1.avg)
            self.logs['test_loss_trans_latent'].append(test_loss4.avg)
            self.logs['test_loss_pde'].append(test_loss5.avg)
        
        print('# {} {:1.3f} | loss1: {:1.2e}  loss2: {:1.2e}  loss3: {:1.2e} loss4: {:1.2e} loss5: {:1.2e} | loss1: {:1.2e} loss2: {:1.2e}  loss3: {:1.2e} loss4: {:1.2e} loss5: {:1.2e}'
              .format(epoch, t2-t1, train_loss1.avg, train_loss2.avg, train_loss3.avg, train_loss4.avg, train_loss5.avg, test_loss1.avg, test_loss2.avg, test_loss3.avg, test_loss4.avg, test_loss5.avg))

    def process(self, train_loader, test_loader):
        for epoch in range(1, self.epochs+1):
            self.train_test(epoch, train_loader, test_loader)


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
