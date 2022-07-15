import datetime
import os
import time

import tools.csv_exporter as csv_exporter
import tools.new_plotter as new_plotter

## new lenet e=50
RESULT_CSV_1 = '/home/lylin1997/repos/pytorch-cifar/results/06-29-2022_123553_lenet_e=50_pre=0.1_window=5_staticTrue_gfTrue/results_lenet_e=50_pre=0.1_window=5_staticTrue_gfTrue.csv'
RESULT_CSV_2 = '/home/lylin1997/repos/pytorch-cifar/results/06-29-2022_123611_lenet_e=50_pre=0.25_window=5_staticTrue_gfTrue/results_lenet_e=50_pre=0.25_window=5_staticTrue_gfTrue.csv'
METRIC_CSV_1 = '/home/lylin1997/repos/pytorch-cifar/results/06-29-2022_123553_lenet_e=50_pre=0.1_window=5_staticTrue_gfTrue/metrics_lenet_e=50_pre=0.1_window=5_staticTrue_gfTrue.csv'
METRIC_CSV_2 = '/home/lylin1997/repos/pytorch-cifar/results/06-29-2022_123611_lenet_e=50_pre=0.25_window=5_staticTrue_gfTrue/metrics_lenet_e=50_pre=0.25_window=5_staticTrue_gfTrue.csv'

## new mobilenet e=100
# RESULT_CSV_1 = '/home/lylin1997/repos/pytorch-cifar/results/06-28-2022_181730_mobilenet_e=100_pre=0.1_window=5_staticTrue_gfTrue/results_mobilenet_e=100_pre=0.1_window=5_staticTrue_gfTrue.csv'
# RESULT_CSV_2 = '/home/lylin1997/repos/pytorch-cifar/results/06-29-2022_123627_mobilenet_e=100_pre=0.25_window=5_staticTrue_gfTrue/results_mobilenet_e=100_pre=0.25_window=5_staticTrue_gfTrue.csv'
# METRIC_CSV_1 = '/home/lylin1997/repos/pytorch-cifar/results/06-28-2022_181730_mobilenet_e=100_pre=0.1_window=5_staticTrue_gfTrue/metrics_mobilenet_e=100_pre=0.1_window=5_staticTrue_gfTrue.csv'
# METRIC_CSV_2 = '/home/lylin1997/repos/pytorch-cifar/results/06-29-2022_123627_mobilenet_e=100_pre=0.25_window=5_staticTrue_gfTrue/metrics_mobilenet_e=100_pre=0.25_window=5_staticTrue_gfTrue.csv'


## new resnet e==100
# RESULT_CSV_1 = '/home/lylin1997/repos/pytorch-cifar/results/06-28-2022_181718_resnet_e=100_pre=0.1_window=5_staticTrue_gfTrue/results_resnet_e=100_pre=0.1_window=5_staticTrue_gfTrue.csv'
# RESULT_CSV_2 = '/home/lylin1997/repos/pytorch-cifar/results/06-28-2022_181737_resnet_e=100_pre=0.25_window=5_staticTrue_gfTrue/results_resnet_e=100_pre=0.25_window=5_staticTrue_gfTrue.csv'
# METRIC_CSV_1 = '/home/lylin1997/repos/pytorch-cifar/results/06-28-2022_181718_resnet_e=100_pre=0.1_window=5_staticTrue_gfTrue/metrics_resnet_e=100_pre=0.1_window=5_staticTrue_gfTrue.csv'
# METRIC_CSV_2 = '/home/lylin1997/repos/pytorch-cifar/results/06-28-2022_181737_resnet_e=100_pre=0.25_window=5_staticTrue_gfTrue/metrics_resnet_e=100_pre=0.25_window=5_staticTrue_gfTrue.csv'

def setup_folders(model_type):
    base_dir = os.path.dirname(__file__)
    now = datetime.datetime.now()
    dt_string = now.strftime("%m-%d-%Y_%H%M%S")
    results_dir = os.path.join(base_dir, 'results/best_acc', f'{dt_string}_{model_type}')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    print(results_dir)
    return results_dir

def plot_best_acc(output_dir=None, model_type=None):
    csv_file_1 = RESULT_CSV_1
    csv_file_2 = RESULT_CSV_2
    
    all_data = []
    all_data.append(csv_exporter.import_csv(filepath=csv_file_1))
    all_data.append(csv_exporter.import_csv(filepath=csv_file_2))
    # model_type = 'resnet' if 'resnet' in csv_file_1 else 'mobilenet'
    # model_type = 'mobilenet' if 'mobilenet' in csv_file_1 else 'lenet'

    new_plotter.plot_best_acc(all_data=all_data, title=f'Best Accuracy w.r.t.Initial Freezing Point (Model: {model_type})', figure_idx=2)
    new_plotter.save_figure(os.path.join(output_dir, f'{model_type}_best_acc.png'))  


def plot_total_training_time(output_dir=None, model_type=None):
    csv_file_1 = METRIC_CSV_1
    csv_file_2 = METRIC_CSV_2
    
    raw_data = []
    raw_data.append(csv_exporter.import_csv(filepath=csv_file_1))
    raw_data.append(csv_exporter.import_csv(filepath=csv_file_2))
    
    new_all_data = []
    for data in raw_data:
        new_all_data.append(new_plotter.calc_transmission_speedup(all_data=data))
    print(new_all_data)

    new_plotter.plot_training_time(all_data=new_all_data,  figure_idx=4, model_type=model_type)
    new_plotter.save_figure(os.path.join(output_dir, f'{model_type}_total_training_time.png')) 


def plot_total_trainable_params(output_dir=None, model_type=None):
    csv_file_1 = METRIC_CSV_1
    csv_file_2 = METRIC_CSV_2
    
    raw_data = []
    raw_data.append(csv_exporter.import_csv(filepath=csv_file_1))
    raw_data.append(csv_exporter.import_csv(filepath=csv_file_2))

    new_plotter.plot_transmission_volume(all_data=raw_data,  figure_idx=5, model_type=model_type)
    new_plotter.save_figure(os.path.join(output_dir, f'{model_type}_total_trainable_params.png')) 
    new_plotter.block_show()


if __name__ == '__main__':
    
    if 'resnet' in RESULT_CSV_1:
        model_type = 'resnet'
    elif 'mobilenet' in RESULT_CSV_1:
        model_type = 'mobilenet'
    else:
        model_type = 'lenet'
    output_dir = setup_folders(model_type=model_type)

    plot_best_acc(output_dir=output_dir, model_type=model_type)
    time.sleep(1)
    plot_total_training_time(output_dir=output_dir, model_type=model_type)
    time.sleep(1)
    plot_total_trainable_params(output_dir=output_dir, model_type=model_type)
    time.sleep(1)

    # new_plotter.show()