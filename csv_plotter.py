import datetime
import os
from xxlimited import new
import tools.csv_exporter as csv_exporter
import tools.new_plotter as new_plotter

def setup_folders():
    base_dir = os.path.dirname(__file__)
    now = datetime.datetime.now()
    dt_string = now.strftime("%m-%d-%Y_%H%M%S")
    results_dir = os.path.join(base_dir, 'results/best_acc', f'{dt_string}')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    print(results_dir)
    return results_dir

if __name__ == '__main__':
    output_dir = setup_folders()
    
    # MobileNet Results
    # csv_file_1 = '/home/lylin1997/repos/pytorch-cifar/results/05-25-2022_085722_mobilenet_e=50_pre=0.1_window=5_staticTrue_gfTrue/results_mobilenet_e=50_pre=0.1_window=5_staticTrue_gfTrue.csv'
    # csv_file_2 = './results/05-25-2022_101529_mobilenet_e=50_pre=0.25_window=5_staticTrue_gfTrue/results_mobilenet_e=50_pre=0.25_window=5_staticTrue_gfTrue.csv'
    
    # MobileNet Metrics
    # csv_file_1 = '/home/lylin1997/repos/pytorch-cifar/results/05-25-2022_085722_mobilenet_e=50_pre=0.1_window=5_staticTrue_gfTrue/metrics_mobilenet_e=50_pre=0.1_window=5_staticTrue_gfTrue.csv'
    # csv_file_2 = '/home/lylin1997/repos/pytorch-cifar/results/05-25-2022_101529_mobilenet_e=50_pre=0.25_window=5_staticTrue_gfTrue/metrics_mobilenet_e=50_pre=0.25_window=5_staticTrue_gfTrue.csv'
    
    # ResNet Metrics
    csv_file_1 = '/home/lylin1997/repos/pytorch-cifar/results/05-25-2022_082102_resnet_e=50_pre=0.1_window=5_staticTrue_gfTrue/metrics_resnet_e=50_pre=0.1_window=5_staticTrue_gfTrue.csv'
    csv_file_2 = '/home/lylin1997/repos/pytorch-cifar/results/05-25-2022_044600_resnet_e=50_pre=0.25_window=5_staticTrue_gfTrue/metrics_resnet_e=50_pre=0.25_window=5_staticTrue_gfTrue.csv'

    all_data = []
    all_data.append(csv_exporter.import_csv(filepath=csv_file_1))
    all_data.append(csv_exporter.import_csv(filepath=csv_file_2))
    model_type = 'resnet' if 'resnet' in csv_file_1 else 'mobilenet'

    # new_plotter.plot_best_acc(all_data=all_data, title=f'Best Accuracy w.r.t.Initial Freezing Point (Model: {model_type})', figure_idx=1)
    # new_plotter.save_figure(os.path.join(output_dir, f'{model_type}_best_acc.png'))
    new_plotter.plot_transmission_ratio(all_data=all_data, title=f'Transmission volume reduction w.r.t.Initial Freezing Point (Model: {model_type})', figure_idx=1)
    new_plotter.save_figure(os.path.join(output_dir, f'{model_type}_transmission_volume_reduction.png'))
    
    
    new_plotter.show()