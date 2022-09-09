from scripts.draw_utils import *

if __name__ == '__main__':
    ex_nums = ['ex0', 'ex3', 'ex3_nomod']
    # ex_nums = ['ex0_big', 'ex3_big', 'ex3_big_nomod']
    label = ['without_pde_loss', 'with_modify', 'without_modify']
    draw_loss_plot(ex_nums, label)