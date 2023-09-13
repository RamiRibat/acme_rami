import os, yaml, json
import argparse
import pathlib

import tqdm

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages



def main(args):
    # print('args.logdir: ', args.logdir)
    logdir = args.logdir.expanduser()
    logged_ids = ['evaluator']
    metrics = ['actor_steps', 'episode_return']
    
    pdf = PdfPages(f'{logdir}/results.pdf')

    path = os.path.join(os.getcwd()+'/config.yaml')
    config = yaml.safe_load(open(path))
    control_suite = config['control']
    control_levels = ['trivial', 'easy', 'medium', 'hard']

    info = {}
    agents, suites, levels, tasks, seeds, logged = None, None, None, None, None, None
    base_ptr = str(logdir).count(os.path.sep)

    for root, dirs, files in os.walk(logdir):
        ptr, ids = root.count(os.path.sep), root.rsplit(os.path.sep)

        if ptr == base_ptr:
            if agents is None: agents = dirs
        if ptr == base_ptr+1:
            if suites is None: suites = dirs
            info[ids[-1]] = {}
        if ptr == base_ptr+2:
            # if levels is None: levels = dirs
            info[ids[-2]][ids[-1]] = {}
        if ptr == base_ptr+3:
            # if tasks is None: tasks = dirs # [x.replace('_', ':') for x in dirs]
            info[ids[-3]][ids[-2]][ids[-1]] = {}
        if ptr == base_ptr+4:
            # if seeds is None: seeds = dirs
            info[ids[-4]][ids[-3]][ids[-2]][ids[-1]] = {}
        if ptr == base_ptr+5: # seed level
            if logged is None: logged = ['actor', 'learner', 'evaluator']
            info[ids[-5]][ids[-4]][ids[-3]][ids[-2]][ids[-1]] = {}
        if ptr == base_ptr+6: # logged level
            info[ids[-6]][ids[-5]][ids[-4]][ids[-3]][ids[-2]][ids[-1]] = {'path': None, 'metrics': {}}
            if ids[-1] == 'checkpoints':
                pass
            else:
                path = os.path.join(root, files[0])
                info[ids[-6]][ids[-5]][ids[-4]][ids[-3]][ids[-2]][ids[-1]]['path'] = path
                if ids[-1] in logged_ids:
                    df = pd.DataFrame(pd.read_csv(path))
                    if ids[-1] == 'evaluator':
                        df = df.groupby('actor_steps').mean().reset_index()
                    for metric in metrics:
                        info[ids[-6]][ids[-5]][ids[-4]][ids[-3]][ids[-2]][ids[-1]]['metrics'][metric] = df[metric].to_list()
    
    # print('info: ', info)

    ratios = [1]*(8)
    ratios[0] = 0.5
    # plt.style.use('seaborn-v0_8')
    plt.style.use('seaborn-v0_8-darkgrid')
    # plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(ncols=4, nrows=8, figsize=(8, 12), sharey=True, gridspec_kw={'height_ratios': ratios}, constrained_layout=True) # constrained_layout=True
    fig.suptitle('DMC Episode Return', fontsize='x-large', weight="bold")
    # fig.supxlabel(args.x_axis)
    fig.text(0.9, 0.01, args.x_axis, va='center', ha='center')
    # fig.tight_layout()
    
    for ax, level in zip(axs.flatten()[:4], control_suite.keys()):
        ax.axis('off')
        # ax.set_title(level, loc='lower center')
        ax.text(0.5, 0.5, level, va='center', ha='center', fontsize='large')


    agents, suites, levels, tasks, plots = [], [], [], [], []
    for agent in info.keys():
        agents.append(agent)
        for suite in info[agent].keys():
            suites.append(suite)
            for level in info[agent][suite].keys():
                levels.append(level)
                for task in info[agent][suite][level].keys(): # axs
                    tasks.append(task)
                    X, Y = [], []
                    for seed in info[agent][suite][level][task].keys():
                        x_data = info[agent][suite][level][task][seed]['evaluator']['metrics'][args.x_axis]
                        y_data = info[agent][suite][level][task][seed]['evaluator']['metrics'][args.y_axis]
                        X = x_data if len(x_data) > len(X) else X
                        Y.append(y_data)
                    info[agent][suite][level][task]['results'] = {}
                    info[agent][suite][level][task]['results']['Y_MEAN'] = np.mean(Y, 0)
                    info[agent][suite][level][task]['results']['Y_STD'] = np.std(Y, 0)
                    info[agent][suite][level][task]['results']['X'] = X

            for i in range(4):
                level = control_levels[i]
                TASKS = control_suite[level]['tasks']
                for j in range(1, 8):
                    task = TASKS[j-1]
                    axs[j][i].set_title(task, fontsize='small')
                    axs[j][i].set_ylim([0, 1000])
                    axs[j][i].set_yticks([0, 250, 500, 750, 1000])
                    if i == 0: axs[j][i].set_yticklabels(['', '250', '500', '750', '1000'])
                    
                    formatter = ticker.ScalarFormatter(useMathText=True)
                    formatter.set_scientific(True) 
                    formatter.set_powerlimits((-1,1)) 
                    axs[j][i].xaxis.set_major_formatter(formatter)

                    if j < 7:
                        axs[j][i].set_xticklabels([])
                        
                    if i == 0:
                        axs[j][i].set_xlim([0, 500_000])
                    if i == 1:
                        axs[j][i].set_xlim([0, 1_000_000])
                    if i == 2:
                        axs[j][i].set_xlim([0, 2_000_000])
                    if i == 3:
                        axs[j][i].set_xlim([0, 5_000_000])
                    
                    # print('suite: ', info[agent][suite])
                    if level in levels and task in tasks:
                        X = info[agent][suite][level][task]['results']['X']
                        MEAN = info[agent][suite][level][task]['results']['Y_MEAN']
                        STD = info[agent][suite][level][task]['results']['Y_STD']
                        plot = axs[j][i].plot(X, MEAN, label='d4pg')
                        axs[j][i].fill_between(X, MEAN-STD, MEAN+STD, alpha=0.2)
                        handles, labels = axs[j][i].get_legend_handles_labels()

                    axs[1][i].get_shared_x_axes().join(axs[1][i], axs[j][i])

        # plots.append(plot)
    
    fig.legend(handles, labels, fontsize='x-large', loc='center', bbox_to_anchor=(0.5, 0.95, 0.005, 0.005), frameon=True)

    pdf.savefig(fig)
    pdf.close()

    # plt.subplot_tool()
    # plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=pathlib.Path, required=True)
    parser.add_argument('--x-axis', type=str, default='actor_steps')
    parser.add_argument('--y-axis', type=str, default='episode_return')
    args = parser.parse_args()
    main(args)