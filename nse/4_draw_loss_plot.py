from scripts.draw_utils import *

if __name__ == '__main__':
    # ex_nums = ['ex0', 'ex7', 'ex7_nomod']
    ex_nums = ['ex0', 'ex1_3', 'ex4_3']
    label = ['baseline', '2-step', '1-step']
    ex_nums = ['ex0', 'ex4_3', 'ex4_4']
    label = ['baseline', 'ex4_3', 'ex4_4']
    draw_loss_plot(ex_nums, label)