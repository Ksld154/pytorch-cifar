from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import datetime
import os

def multiplot(all_data, y_label, title, figure_idx):
    ax1 = plt.figure(num=figure_idx, figsize=(8, 6)).gca()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) # integer x-axis

    # plt.figure(num=figure_idx, figsize=(12, 8))
    plt.title(title)
    plt.ylabel(y_label)   # y label
    plt.xlabel("Epochs")  # x label
    font = {'size'   : 12}
    plt.rc('font', **font)

    for data in all_data:
        # print(data)
        plt.plot(data.get('acc'),
                 label=data.get('name'),
                 marker="o",
                 linestyle="-")
    plt.legend(loc='lower right')

def show():
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(10)
    plt.close()


def save_figure(filepath:str):
    plt.savefig(filepath)
    # print(filepath)

def plot_best_acc(all_data, title, figure_idx):
    ax1 = plt.figure(num=figure_idx, figsize=(8, 6)).gca()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) # integer x-axis

    # model_names = [ t['name'] for t in all_data]
    # print(model_names)
    
    all_group_best_accs = []
    for group_data in all_data:
        best_accs = [ max(model['acc'].strip('][').split(', ')) for model in group_data]
        round_best_accs = [round(float(x), 4) for x in best_accs]
        all_group_best_accs.append(round_best_accs)
        print(round_best_accs)
    
    print(all_group_best_accs)
    x = np.arange(len(all_group_best_accs))
    print(x)

    reshape_all_accs = [list(x) for x in zip(*all_group_best_accs)]
    print(reshape_all_accs)

    color_options = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    model_name = [ t['name'] for t in all_data[0]]
    print(model_name)
    for idx, model_accs in enumerate(reshape_all_accs):
        print(model_accs)
        plt.bar(x+0.1*idx, model_accs, color=color_options[idx], width=0.1, label=model_name[idx])
        # plt.xticks(x, model_names)

    plt.ylim([0.7, 0.9])

    plt.title(title)
    plt.ylabel('Accuracy')  
    plt.xlabel('Initial Freezing Timeslot')  
    # font = {'size': 12}
    # plt.rc('font', **font)
    
    plt.xticks(x + 0.2, ('I=0.1', 'I=0.25'))
    plt.legend()


def plot_transmission_ratio(all_data, title, figure_idx):
    ax1 = plt.figure(num=figure_idx, figsize=(8, 6)).gca()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) # integer x-axis

    
    all_group_data = []
    for group_data in all_data:
        # print(model['save_transmission_ratio'])
        tranmission_ratio = [ model['save_transmission_ratio'].strip('%') for model in group_data]
        print(tranmission_ratio)
        round_tranmission_ratio = [round(float(x), 4) for x in tranmission_ratio]
        all_group_data.append(round_tranmission_ratio)
    print(all_group_data)

    x = np.arange(len(all_group_data))
    reshape_all_accs = [list(x) for x in zip(*all_group_data)]
    print(reshape_all_accs)

    color_options = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    model_name = [ t['name'] for t in all_data[0]]
    print(model_name)
    for idx, model_accs in enumerate(reshape_all_accs):
        print(model_accs)
        plt.bar(x+0.1*idx, model_accs, color=color_options[idx], width=0.1, label=model_name[idx])
        # plt.xticks(x, model_names)

    # plt.ylim([0.7, 0.9])

    plt.title(title)
    plt.ylabel('Accuracy')  
    plt.xlabel('Initial Freezing Timeslot')  
    # font = {'size': 12}
    # plt.rc('font', **font)
    
    plt.xticks(x + 0.2, ('I=0.1', 'I=0.25'))
    plt.legend()
