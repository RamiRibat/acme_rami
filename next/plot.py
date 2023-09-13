import os
import argparse
import pathlib

import tqdm

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages




DMC_TASK_NAMES = [
    ['ball_in_cup_catch', 'cartplole_balance', 'cartplole_balance_sparse', 'cartpole_swingup', 'reacher_easy', 'walker_stand', 'walker_walk'],
    ['cartpole_swingup_sparse', 'finger_turn_easy', 'hopper_stand', 'pendulum_swingup', 'point_mass_easy', 'reacher_hard', 'swimmer_swimmer6'],
    ['cheetah_run', 'finger_spin', 'finger_turn_hard', 'fish_swim', 'fish_upright', 'swimmer_swimmer_15', 'walker_run'],
    ['acrobat_swingup', 'acrobat_swingup_sparse', 'hopper_hop', 'humanoid_run', 'humanoid_stand', 'humanoid_walk', 'manipulator_bring_ball']
]



def main(args):
    # print('args.logdir: ', args.logdir)
    logdir = args.logdir.expanduser()
    logged_ids = ['evaluator']
    metrics = ['actor_steps', 'episode_return']
    
    pdf = PdfPages(f'{logdir}/results.pdf')


    info = {}
    agents, tasks, seeds, logged = None, None, None, None
    base_level = str(logdir).count(os.path.sep)

    for root, dirs, files in os.walk(logdir):
        level, ids = root.count(os.path.sep), root.rsplit(os.path.sep)

        if level == base_level:
            if agents is None: agents = dirs
        if level == base_level+1:
            if tasks is None: tasks = dirs # [x.replace('_', ':') for x in dirs]
            info[ids[-1]] = {}
        if level == base_level+2:
            if seeds is None: seeds = dirs
            info[ids[-2]][ids[-1]] = {}
        if level == base_level+3: # seed level
            if logged is None: logged = ['actor', 'learner', 'evaluator']
            info[ids[-3]][ids[-2]][ids[-1]] = {}
        if level == base_level+4: # logged level
            info[ids[-4]][ids[-3]][ids[-2]][ids[-1]] = {'path': None, 'metrics': {}}
            if ids[-1] == 'checkpoints':
                pass
            else:
                path = os.path.join(root, files[0])
                info[ids[-4]][ids[-3]][ids[-2]][ids[-1]]['path'] = path
                if ids[-1] in logged_ids:
                    df = pd.DataFrame(pd.read_csv(path))
                    if ids[-1] == 'evaluator':
                        # print('evaluator.df: ', df)
                        df = df.groupby('actor_steps').mean().reset_index()
                        # print('evaluator.df(mean): ', df)
                    for metric in metrics:
                        # print(f'df[{metric}]: ', df[metric])
                        info[ids[-4]][ids[-3]][ids[-2]][ids[-1]]['metrics'][metric] = df[metric].to_list()
                        

    ratios = [1]*(8)
    ratios[0] = 0.02
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axs = plt.subplots(ncols=4, nrows=8, figsize=(8, 12), constrained_layout=True, gridspec_kw={'height_ratios': ratios})
    fig.suptitle('DMC Episode Return', weight="bold")
    fig.supxlabel(args.x_axis)
    
    for ax, level in zip(axs.flatten()[:4], ['trivial', 'easy', 'medium', 'hard']):
        ax.axis('off')
        ax.set_title(level)

    for agent in agents:

        for task in tasks: # axs
            X, Y = [], []
            info[agent][task]['results'] = {}
            for seed in seeds:
                # print('seed: ', seed)
                x_data = info[agent][task][seed]['evaluator']['metrics'][args.x_axis]
                y_data = info[agent][task][seed]['evaluator']['metrics'][args.y_axis]
                X = x_data if len(x_data) > len(X) else X
                # print('y_data:', y_data)
                Y.append(y_data)
            info[agent][task]['results']['Y_MEAN'] = np.mean(Y, 0)
            info[agent][task]['results']['Y_STD'] = np.std(Y, 0)
            info[agent][task]['results']['X'] = X

        for i in range(4):
            for j in range(1, 8):
                task = DMC_TASK_NAMES[i][j-1]
                axs[j][i].set_title(task, fontsize='small')
                axs[j][i].set_yticks([0, 250, 500, 750, 1000])
                
                formatter = ticker.ScalarFormatter(useMathText=True)
                formatter.set_scientific(True) 
                formatter.set_powerlimits((-1,1)) 
                axs[j][i].xaxis.set_major_formatter(formatter)

                if j < 7:
                    axs[j][i].set_xticklabels([])
                if i > 0:
                    axs[j][i].set_yticklabels([])
                else:
                    axs[j][i].set_yticklabels(['', '250', '500', '750', '1000'])
                if i == 0:
                    axs[j][i].set_xlim(left=0, right=500_000)
                if i == 1:
                    axs[j][i].set_xlim(left=0, right=1_000_000)
                if i == 2:
                    axs[j][i].set_xlim(left=0, right=2_000_000)
                if i == 3:
                    axs[j][i].set_xlim(left=0, right=5_000_000)

                if f'control_{task}' in tasks:
                    task = f'control_{task}'
                    X = info[agent][task]['results']['X']
                    MEAN = info[agent][task]['results']['Y_MEAN']
                    STD = info[agent][task]['results']['Y_STD']
                    axs[j][i].plot(X, MEAN)
                    axs[j][i].fill_between(X, MEAN-STD, MEAN+STD, alpha=0.2)


    pdf.savefig(fig)
    pdf.close()

    # plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=pathlib.Path, required=True)
    parser.add_argument('--x-axis', type=str, default='actor_steps')
    parser.add_argument('--y-axis', type=str, default='episode_return')
    args = parser.parse_args()
    main(args)