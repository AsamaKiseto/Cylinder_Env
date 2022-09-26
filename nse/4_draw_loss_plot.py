from scripts.draw_utils import *
from scripts.utils import *
from scripts.nse_model import *

# fig setting
fig, ax = plt.subplots(nrows=6, ncols=2, figsize=(15,12), dpi=1000)
ax = ax.flatten()
# plt.figure(figsize=(15, 12))
for i in range(6):
    ax[i] = plt.subplot2grid((6, 2), (i, 0), colspan=2)
    ax[i].grid(True, lw=0.4, ls="--", c=".50")
    ax[i].set_yscale('log')
    # ax[i].set_ylabel(f'loss{i+1}', fontsize=15)
    ax[i].set_ylim(1e-4, 1)

ax[0].set_title("Loss Plots", fontsize=10)
ax[0].set_ylabel("pred loss", fontsize=10)
ax[1].set_ylabel("recon loss of state", fontsize=10)
ax[2].set_ylabel("recon loss of", fontsize=10)
ax[3].set_ylabel("latent loss", fontsize=10)
ax[4].set_ylabel("phys loss of obs", fontsize=10)
ax[5].set_ylabel("phys loss of pred", fontsize=10)
ax[5].set_xlabel("epochs", fontsize=10)

if __name__ == '__main__':
    print('start load data')

    data = LoadData('data/nse_data_reg')
    ex_nums = ['ex0', 'ex1', 'ex4']
    label = ['baseline', '2-step', '1-step']
    color = ['black', 'blue', 'yellow']

    N = len(ex_nums)
    print(ex_nums)
    _, _, logs_base = torch.load(f"logs/phase1_{ex_nums[0]}_grid_pi")
    args, data_norm, _ = logs_base['args'], logs_base['data_norm'], logs_base['logs']

    data.split(args.Ng, args.tg)
    data.norm()
    data_loader = data.trans2CheckSet(args.batch_size)
    _, data_loader = data.trans2TrainingSet(args.batch_size)
    N0, nt, nx, ny = data.get_params()
    shape = [nx, ny]
    
    for k in range(len(ex_nums)):
        print(ex_nums[k], label[k])

        _, _, logs = torch.load(f"logs/phase1_{ex_nums[k]}_grid_pi")
        logs = logs['logs']
        model = NSEModel_FNO(args, shape, data.dt)
        pred_model, phys_model = logs['pred_model'], logs['phys_model']
        epochs = len(pred_model)
        loss = np.zeros((6, epochs))

        print('begin simulation')
        for i in range(epochs):
            print(f'# {i+1}')
            model.load_state(pred_model[i], phys_model[i])
            loss[0, i], loss[1, i], loss[2, i], loss[3, i], loss[4, i], loss[5, i] = model.simulate(data_loader)
        print('end simulation')
        
        for i in range(6):
            ax[i].plot(loss[i], color=color[k], label=label[k])
            ax[i].legend()

    plt.savefig('logs/loss_plot.jpg')