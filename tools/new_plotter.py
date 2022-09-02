import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import datetime
import os


DEFAULT_FIGURE_SIZE = 2
color_options  = ['r', 'g', 'b', 'c', 'y', 'k']
linestyle_options = ['-', '--', '-.', ':']
marker_options = ['o', '*', '.', ',', 'x', 'P', 'D', 'H']
hatch_options = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']

width = 0.15

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

def plot_acc(all_data, figure_idx):
    ax1 = plt.figure(num=figure_idx, figsize=(8, 6)).gca()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) # integer x-axis

    plt.title('')
    plt.ylabel('Accuracy')   # y label
    plt.xlabel("Epochs")  # x label
    font = {'size'   : 12}
    plt.rc('font', **font)

    for data in all_data:
        # print(len(data.get('acc')))

        acc = eval(data.get('acc'))
        print(len(acc))
        plt.plot(acc,
                 label=data.get('name'),
                 marker="o",
                 linestyle="-")
    plt.legend(loc='lower right')

def show():
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(5)
    plt.close()


def block_show():
    plt.gcf().subplots_adjust(left=0.15)
    plt.tight_layout()
    plt.show()
    # plt.pause(5)
    # plt.close()

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
        plt.bar(x+width*idx, model_accs, color=color_options[idx], width=width, label=model_name[idx])
        # plt.xticks(x, model_names)

    plt.ylim([0, 1])

    plt.title(title)
    plt.ylabel('Accuracy')  
    plt.xlabel('Initial Freezing Timeslot')  
    # font = {'size': 12}
    # plt.rc('font', **font)
    
    plt.xticks(x + 0.2, ('I=width', 'I=0.25'))
    plt.legend(loc='lower right')


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
        plt.bar(x+width*idx, model_accs, color=color_options[idx], width=width, label=model_name[idx])
        # plt.xticks(x, model_names)

    # plt.ylim([0.7, 0.9])

    plt.title(title)
    plt.ylabel('Transmission Volume Reduce')  
    plt.xlabel('Initial Freezing Timeslot')  
    # font = {'size': 12}
    # plt.rc('font', **font)
    
    plt.xticks(x + 0.2, ('I=0.1', 'I=0.25'))
    plt.legend(loc='lower right')

def calc_transmission_speedup(all_data):
    baseline_time = 0
    result_all_data = []
    for idx, data in enumerate(all_data):
        if data.get('total_training_time'):
            pt = datetime.datetime.strptime(data['total_training_time'],'%H:%M:%S.%f')
            tt = pt - datetime.datetime(1900, 1, 1)
            total_seconds = tt.total_seconds()
            print(total_seconds)
            data['total_training_time'] = total_seconds
            
            if 'Baseline' in data['name']:
                baseline_time = total_seconds
                print(baseline_time)
            
            
    for data in all_data:
        data['speedup_ratio'] = baseline_time / data['total_training_time']
        result_all_data.append(data)
    print(result_all_data)
    return result_all_data


def plot_speedup_ratio(all_data, title, figure_idx):
    ax1 = plt.figure(num=figure_idx, figsize=(8, 6)).gca()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) # integer x-axis

    all_group_data = []
    for group_data in all_data:
        speedup_ratio = [ round(float(model['speedup_ratio']), 4) for model in group_data]
        print(speedup_ratio)
        # round_tranmission_ratio = [round(float(x), 4) for x in tranmission_ratio]
        all_group_data.append(speedup_ratio)
    print(all_group_data)


    x = np.arange(len(all_group_data))
    reshape_all_group_data = [list(x) for x in zip(*all_group_data)]
    print(reshape_all_group_data)

    color_options = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    model_name = [ t['name'] for t in all_data[0]]
    print(model_name)
    for idx, model_speedup in enumerate(reshape_all_group_data):
        print(model_speedup)
        plt.bar(x+width*idx, model_speedup, color=color_options[idx], width=width, label=model_name[idx])
        # plt.xticks(x, model_names)

    plt.title(title)
    plt.ylabel('Speedup')  
    plt.xlabel('Initial Freezing Timeslot')  
    plt.xticks([])
    # font = {'size': 12}
    # plt.rc('font', **font)
    
    plt.xticks(x + 0.2, ('I=0.1', 'I=0.25'))
    plt.legend(loc='lower right')


def plot_training_time(all_data, figure_idx, model_type):
    font = {'size'   : 14}
    plt.rc('font', **font)
    ax1 = plt.figure(num=figure_idx, figsize=(6, 6)).gca()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) # integer x-axis
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    all_group_data = []
    for group_data in all_data:
        speedup_ratio = [ float(model['total_training_time']) for model in group_data]
        # print(speedup_ratio)
        all_group_data.append(speedup_ratio)
    print(all_group_data)

    x = np.arange(len(all_group_data))
    reshape_all_group_data = [list(x) for x in zip(*all_group_data)]
    print(reshape_all_group_data)

    if model_type == 'resnet':
        color_options  = ['r', 'g', 'b', 'c', 'y', 'k']
        hatch_options = ['/', '\\', '|', '-', 'x', 'o', 'O', '.', '*']
    else:
        color_options  = ['r', 'g', 'b', 'c', 'y', 'k']
        linestyle_options = ['-', '--', '-.', ':']
        marker_options = ['o', '*', '.', ',', 'x', 'P', 'D', 'H']
        hatch_options = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']


    model_name = [ t['name'] for t in all_data[0]]
    print(model_name)
    for idx, model_speedup in enumerate(reshape_all_group_data):
        print(model_speedup)
        plt.bar(x+width*idx, model_speedup,  color=color_options[idx % len(color_options)], hatch=hatch_options[idx % len(hatch_options)], width=width, label=model_name[idx])

    plt.ylabel('Training Duration (seconds)')  
    plt.xlabel('Initial Freezing Timeslot')  
    plt.xticks([])
    
    if model_type == 'resnet':
        plt.xticks(x + 0.2, ('I=0.1', 'I=0.25'))
    else:
        plt.xticks(x + 0.25, ('I=0.1', 'I=0.25'))

    # leg = plt.legend(frameon=True, loc='lower right')
    # leg.set_draggable(state=True)     
    if model_type == 'resnet':
        leg = plt.legend(frameon=True, loc='lower right')
        leg.set_draggable(state=True)    

def plot_transmission_volume(all_data, figure_idx, model_type):
    font = {'size'   : 14}
    plt.rc('font', **font)
    ax1 = plt.figure(num=figure_idx, figsize=(6, 6)).gca()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) # integer x-axis
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    all_group_data = []
    for group_data in all_data:
        speedup_ratio = [ float(model['total_trainable_weights'])*4/1024/1024 for model in group_data]
        all_group_data.append(speedup_ratio)
    print(all_group_data)

    x = np.arange(len(all_group_data))
    reshape_all_group_data = [list(x) for x in zip(*all_group_data)]
    print(reshape_all_group_data)

    if model_type == 'resnet':
        color_options  = ['r', 'g', 'b', 'c', 'y', 'k']
        hatch_options = ['/', '\\', '|', '-', 'x', 'o', 'O', '.', '*']
    else:
        color_options  = ['r', 'g', 'b', 'c', 'y', 'k']
        linestyle_options = ['-', '--', '-.', ':']
        marker_options = ['o', '*', '.', ',', 'x', 'P', 'D', 'H']
        hatch_options = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']

    model_name = [ t['name'] for t in all_data[0]]
    print(model_name)
    for idx, model_speedup in enumerate(reshape_all_group_data):
        print(model_speedup)
        plt.bar(x+width*idx, model_speedup,  color=color_options[idx % len(color_options)], hatch=hatch_options[idx % len(hatch_options)], width=width, label=model_name[idx])

    plt.ylabel('Trainable Parameters Size (MB)')  
    plt.xlabel('Initial Freezing Timeslot')  
    plt.xticks([])
    # plt.ylim(top=1200)
    
    if model_type == 'resnet':
        plt.xticks(x + 0.2, ('I=0.1', 'I=0.25'))
    else:
        plt.xticks(x + 0.25, ('I=0.1', 'I=0.25'))
    
    # leg = plt.legend(frameon=True, loc='lower right')
    # leg = plt.legend(frameon=False)
    if model_type == 'resnet':
        leg = plt.legend(frameon=True, loc='lower right')
        leg.set_draggable(state=True)    
