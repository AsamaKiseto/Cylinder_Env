# from scripts.draw_utils import *
import matplotlib.pyplot as plt 
from scripts.utils import *
from nse.recycle.nse_model_test import *

ex_nums = ['data_based_bn', 'baseline_bn']
label = ['data_based_bn', 'baseline_bn']

# fig setting
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(15,12), dpi=500)
ax = ax.flatten()
# plt.figure(figsize=(15, 12))
for i in range(3):
    ax[i] = plt.subplot2grid((3, 1), (i, 0))
    ax[i].grid(True, lw=0.4, ls="--", c=".50")
    ax[i].set_yscale('log')
    # ax[i].set_ylabel(f'loss{i+1}', fontsize=15)
    ax[i].set_ylim(1e-4, 1)

ax[0].set_title("loss plot", fontsize=10)
ax[0].set_ylabel("pred loss", fontsize=10)
ax[1].set_ylabel("phys loss of obs", fontsize=10)
ax[2].set_ylabel("phys loss of pred", fontsize=10)
ax[2].set_xlabel("epochs", fontsize=10)

if __name__ == '__main__':
    print('start load data')
    data = LoadData('data/nse_data_reg_dt_0.01_fr_1.0')
    # data = LoadData('data/test_data/nse_data_reg_dt_0.01_fb_0.0_scale_0.1')
    print('finish load date')

    N = len(ex_nums)
    print(ex_nums)
    _, _, logs_base = torch.load(f"logs/phase1_{ex_nums[0]}_grid_pi")
    args, data_norm = logs_base['args'], logs_base['data_norm']

    data.split(args.Ng, args.tg)
    data.normalize('logs_unif', data_norm)
    data_loader = data.trans2CheckSet(0.1, args.batch_size)
    # _, data_loader = data.trans2TrainingSet(args.batch_size)
    N0, nt, nx, ny = data.get_params()
    shape = [nx, ny]
    
    for k in range(len(ex_nums)):
        print(ex_nums[k], label[k])

        _, _, logs = torch.load(f"logs/phase1_{ex_nums[k]}_grid_pi")
        model = NSEModel_FNO(shape, data.dt, args)
        pred_model, phys_model = logs['pred_model'], logs['phys_model']
        epochs = len(pred_model)
        print(f'epochs: {epochs}')
        loss = np.zeros((3, epochs))

        print('begin simulation')
        for i in range(epochs):
            t1 = default_timer()
            model.load_state(pred_model[i], phys_model[i])
            loss[0, i], _, _, _, loss[1, i], loss[2, i] = model.simulate(data_loader)
            t2 = default_timer()
            print(f'# {i+1} : {t2 - t1} | {loss[0, i].mean()} | {loss[1, i].mean()} | {loss[2, i].mean()}')
        print('end simulation')
        
        for i in range(3):
            ax[i].plot(loss[i], label=label[k])
            ax[i].legend()
        
        torch.save(loss, 'logs/data/loss_log_'+ex_nums[k])

    plt.savefig('logs/loss_plot_bn.jpg')