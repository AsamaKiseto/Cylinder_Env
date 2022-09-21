from scripts.draw_utils import *

if __name__ == '__main__':
    # ex_nums = ['ex0', 'ex7', 'ex7_nomod']
    ex_nums = ['ex0', 'ex3_2', 'ex4_2']
    label = ['baseline', '2-step', '1-step']
    draw_loss_plot(ex_nums, label)