import os
import datetime
import time
import copy

import numpy as np
import tabulate
import torchsummary 

import utils
from trainer import Trainer
from constants import *
import tools.new_plotter
import tools.csv_exporter
import tools.options
from tools.utils import moving_average
from tools.data_preprocessing import data_preprossing
from model import MyModel


class Client():
    def __init__(self, cmd_args) -> None:
        self.args = cmd_args

        self.epochs = 10
        self.pre_epochs = cmd_args.pre_epochs_ratio * self.epochs
        self.loss_delta = []

        self.all_trainers = []
        self.all_results = []
        self.all_metrics = []

        self.results_dir = None
        self.base_filename = None


    def train_process(self):
        print(self.args.lr)

        self.epochs = self.args.epochs
        self.pre_epochs = int(self.args.pre_epochs_ratio * self.epochs)
        print(f'Total Epochs: {self.epochs}, Pre-Epochs: {self.pre_epochs}')

        self.setup_folders()
        pretrained_trainer = self.setup_and_pretrain(model_type=self.args.model)
        
        gf1 = None
        if self.args.gradually_freeze:
            gf1 = GraduallyFreezing(pretrained_trainer=pretrained_trainer, pre_epochs=self.pre_epochs)
            primary_results, _ = gf1.train_process(cmd_args=self.args)
            self.all_results.append(primary_results)

        if self.args.static_freeze:
            self.train_all_static_freeze_model(pretrained_trainer=pretrained_trainer)
            
            if self.args.gradually_freeze:
                self.all_metrics = self.print_metrics(gf_primary_trainer=gf1.primary_trainer)
            else:
                self.all_metrics = self.print_metrics(gf_primary_trainer=None)
                

    def setup_folders(self):
        self.base_filename = f"{self.args.model}_e={self.args.epochs}_pre={self.args.pre_epochs_ratio}_window={self.args.window_size}_static{self.args.static_freeze}_gf{self.args.gradually_freeze}"
        print(self.base_filename)
        
        base_dir = os.path.dirname(__file__)
        now = datetime.datetime.now()
        dt_string = now.strftime("%m-%d-%Y_%H%M%S")
        results_dir = os.path.join(base_dir, 'results/', f'{dt_string}_{self.base_filename}')
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        print(results_dir)
        self.results_dir = results_dir

    def setup_and_pretrain(self, model_type):
        # do pre-training before freezing

        train_loader, test_loader = data_preprossing()
        primary_model = MyModel(args=self.args, trainloader=train_loader, testloader=test_loader, model_name=model_type)
        # print(model_type)

        # torchsummary.summary(primary_model.net, (3,28,28), device=primary_model.device)
        primary_trainer = Trainer(model=primary_model, freeze_idx=0, old_obj=False)
        print(primary_trainer.model_type)
        # print(primary_trainer.freeze_options)
        # primary_trainer.model_type = model_type

        
        for e in range(self.pre_epochs):
            print(f'[Pre-Training Epoch {e+1}/{self.pre_epochs}]')
            primary_trainer.train_and_test_epoch(e+1)
        return primary_trainer



    def train_all_static_freeze_model(self, pretrained_trainer):
        for i in range(self.args.static_freeze_candidates):
            if i*pretrained_trainer.freeze_step_size >= len(pretrained_trainer.model.net.layers):
                break

            new_trainer = pretrained_trainer.static_freeze(freeze_degree=i*pretrained_trainer.freeze_step_size)
            new_trainer.name = f'Static Freeze: {i*pretrained_trainer.freeze_step_size} blocks'
            if i == 0:
                new_trainer.name = 'Baseline: No Freeze'
            new_trainer.accuracy = copy.deepcopy(pretrained_trainer.accuracy)
            new_trainer.model.summary()
            self.all_trainers.append(new_trainer)
            

        all_acc = []
        for t in self.all_trainers:
            print(f'Static freeze degree: {t.freeze_idx}')
            for e in range(self.epochs-self.pre_epochs):
                print(f'[Training Epoch {e+1}/{self.epochs}]')
                t.train_and_test_epoch(e+1)
                t.calc_frozen_ratio()
            print(t.accuracy)
            all_acc.append(t.accuracy)

            d = dict(name=t.name, acc=t.accuracy)
            self.all_results.append(d)

        print(all_acc)
        print(self.all_results)

    
    def plot_figure(self):
        tools.new_plotter.multiplot(
            all_data=self.all_results, 
            y_label='Accuracy', 
            title= f'Single-Machine Gradually Freezing Accuracy ({self.args.model} on CIFAR10 dataset)',
            figure_idx=1
        )
        
        png_file = os.path.join(self.results_dir, f'{self.base_filename}.png')
        print(png_file)
        tools.new_plotter.save_figure(png_file)
        tools.new_plotter.show()

    def print_metrics(self, gf_primary_trainer=None):
        table_header = ['Model', 'Training time', 'Transmission parameter volume', 'Save Transmission Ratio']
        table_data = []
        total_transmitted_params = self.all_trainers[0].total_trainable_weights
        
        all_metrics = [] # for output csv

        if gf_primary_trainer:  
            table_data.append(
                (gf_primary_trainer.name, gf_primary_trainer.total_training_time, gf_primary_trainer.total_trainable_weights, 
                f'{ (1 - (gf_primary_trainer.total_trainable_weights/total_transmitted_params) ) * 100} %')
            )
            all_metrics.append(
                dict(
                    name=gf_primary_trainer.name, 
                    total_training_time=gf_primary_trainer.total_training_time, 
                    total_trainable_weights=gf_primary_trainer.total_trainable_weights, 
                    save_transmission_ratio=f'{ (1 - (gf_primary_trainer.total_trainable_weights/total_transmitted_params) ) * 100} %'
                )
            )

        # Static Freezing Metrics
        for trainer in self.all_trainers:
            table_data.append(
                (trainer.name, trainer.total_training_time, trainer.total_trainable_weights, 
                f'{ (1-(trainer.total_trainable_weights/total_transmitted_params)) *100 } %')
            )

            all_metrics.append(
                dict(
                    name=trainer.name, 
                    total_training_time=trainer.total_training_time, 
                    total_trainable_weights=trainer.total_trainable_weights, 
                    save_transmission_ratio=f'{ (1 - (trainer.total_trainable_weights/total_transmitted_params) ) * 100} %'
                )
            )

        print(tabulate.tabulate(table_data, headers=table_header, tablefmt='grid'))

        return all_metrics

    def output_csv(self, data, filename, fields):
        csv_file = os.path.join(self.results_dir, filename)
        print(csv_file)
        tools.csv_exporter.export_csv(data=data, filepath=csv_file, fields=fields)



class GraduallyFreezing():
    def __init__(self, pretrained_trainer=None, pre_epochs=5) -> None:
        # self.data = CIFAR10(BATCH_SIZE)

        self.layer_dicisions = []
        self.loss_delta = []

        self.epochs = 10
        self.pre_epochs = pre_epochs

        self.primary_trainer, self.secondary_trainer = self.setup_and_pretrain(pretrained_trainer=pretrained_trainer)
        self.total_time = None



    def train_process(self, cmd_args=None, transmission_overlap=False, transmission_time=0, switch_model_flag=True):

        table_header = ['Flags', 'Status']
        table_data = [
            ('Overlap', transmission_overlap),
            ('Switch model', switch_model_flag),
            ('Total epochs', cmd_args.epochs),
            ('Window size', cmd_args.window_size),
        ]
        print(tabulate.tabulate(table_data, headers=table_header, tablefmt='grid'))

        self.epochs = cmd_args.epochs
        both_converged = False

        primary_trainer, secondary_trainer = self.primary_trainer, self.secondary_trainer
        primary_trainer.name = f"Primary (degree={primary_trainer.freeze_idx})"

        # In each training epochs
        for e in range(self.epochs-self.pre_epochs):
        
            secondary_trainer = primary_trainer.generate_secondary_trainer(secondary_trainer)
        
            print(f'[Epoch {(e+1)}/{self.epochs}] Base freeze layers: {primary_trainer.freeze_idx}')
            print(f'[Epoch {(e+1)}/{self.epochs}] Next freeze layers: {secondary_trainer.freeze_idx}')
            
            loss_1, _ = primary_trainer.train_and_test_epoch(e+1)

            # train target model first,
            # simulate transmission after target model training is finished
            loss_2, _ = secondary_trainer.train_and_test_epoch(e+1)
            
            self.layer_dicisions.append(primary_trainer.freeze_idx)
            self.loss_delta.append(loss_2 - loss_1)
            models_loss_diff = moving_average(self.loss_delta, cmd_args.window_size)
            print(f'Current Loss Diff.: {loss_2-loss_1}')
            print(f'Avg Loss Difference: {models_loss_diff}')

            # Update transmitted parameter amount
            primary_trainer.calc_frozen_ratio()
            secondary_trainer.calc_frozen_ratio()
        

            if primary_trainer.is_converged(self.pre_epochs, cmd_args.window_size) and secondary_trainer.is_converged(self.pre_epochs, cmd_args.window_size):
                print('*** Both model are converged! ***')
                both_converged = True
        

            # Switch to new model
            if switch_model_flag and both_converged:
                
                # boundary check
                if secondary_trainer.freeze_idx > len(secondary_trainer.model.net.layers) -1 or primary_trainer.freeze_idx >= len(primary_trainer.model.net.layers) -1:
                    continue
                
                # Switch model using Loss difference
                if not np.isnan(models_loss_diff) and models_loss_diff <= LOSS_DIFF_THRESHOLD:
                    print(f'Loss Diff.: {models_loss_diff}, is smaller than threshold, which means model#2 is better')
                    print(f'Loss Diff.: {models_loss_diff}, we will copy model#2 to model#1')

                    self.loss_delta.clear()
                    # Approach 1
                    # secondary_trainer.freeze_idx = primary_trainer.freeze_idx
                    # primary_trainer, _ = self.switch_model_old(primary_trainer, secondary_trainer)
                    
                    # Approach 2
                    primary_trainer = self.switch_model_new(primary_trainer, secondary_trainer)

                    # Approach 3
                    # primary_trainer = primary_trainer.further_freeze(self.pre_epochs, True)
                    # 


        print(self.layer_dicisions)
        print(primary_trainer.accuracy)
        print(secondary_trainer.accuracy)
        print(primary_trainer.loss)
        print(secondary_trainer.loss)
        print(primary_trainer.layer_history)
        print(secondary_trainer.layer_history)
        print(primary_trainer.total_training_time)
        print(secondary_trainer.total_training_time)


        self.primary_trainer = primary_trainer
        self.secondary_trainer = secondary_trainer

        primary_trainer.name = 'Gradually Freezing: Primary Model'
        secondary_trainer.name = "Gradually Freezing: Secondary Model"
        primary_results = dict(name=primary_trainer.name, acc=primary_trainer.accuracy, total_time=str(primary_trainer.total_training_time), freeze_degree=primary_trainer.layer_history)
        secondary_results = dict(name=secondary_trainer.name, acc=secondary_trainer.accuracy, total_time=str(secondary_trainer.total_training_time), freeze_degree=secondary_trainer.layer_history)

        return primary_results, secondary_results


    def setup_and_pretrain(self, pretrained_trainer=False):
        
        primary_trainer = None
        secondary_trainer = None
        if pretrained_trainer:
            primary_trainer = pretrained_trainer.static_freeze(0)
            primary_trainer.name = f'Gradually Freezing: Primary Model'
            primary_trainer.accuracy = copy.deepcopy(pretrained_trainer.accuracy)
            # primary_trainer.model.summary()
            
            # Initailize target_model with all-layers pre-trained base model
            secondary_trainer = pretrained_trainer.static_freeze(1)
            secondary_trainer.name = "Gradually Freezing: Secondary Model"
            secondary_trainer.accuracy = copy.deepcopy(pretrained_trainer.accuracy)
            # secondary_trainer.model.summary()
        
            
        self.primary_trainer = primary_trainer
        self.secondary_trainer = secondary_trainer

        # secondary_trainer act as a dummy trainer here
        return primary_trainer, secondary_trainer


    def switch_model_new(self, primary_trainer, secondary_trainer):

        # New primary model == current "secondary model", since secondary_model is better
        # New secondary model will be generated by primary_trainer.generate_secondary_trainer() at next epoch
        primary_trainer.freeze_idx = secondary_trainer.freeze_idx
        primary_trainer.model.net = copy.deepcopy(secondary_trainer.model.net)
        
        new_primary_trainer = Trainer(
            model=primary_trainer.model,
            freeze_idx=primary_trainer.freeze_idx,
            old_obj=primary_trainer)
        new_primary_trainer.name = f"Primary (degree={new_primary_trainer.freeze_idx})"
        new_primary_trainer.model.summary()
        
        return new_primary_trainer


    def print_metrics(self):
        table_header = ['Model', 'Training time', 'Transmission parameter volume']
        table_data = [
            (self.primary_trainer.name, self.primary_trainer.total_training_time, self.primary_trainer.total_trainable_weights),
            (self.secondary_trainer.name, self.secondary_trainer.total_training_time, self.secondary_trainer.total_trainable_weights),
        ]
        print(tabulate.tabulate(table_data, headers=table_header, tablefmt='grid'))




if __name__ == '__main__':
    args = tools.options.opt_parser()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)

    print('*** Start Training! ***')
    print(f'GPU Device: {args.gpu_device}')
    
    start = time.time()
    client_1 = Client(cmd_args=args)
    client_1.train_process()
    end = time.time()
    print(f'Total training time: {datetime.timedelta(seconds= end-start)}')


    client_1.output_csv(client_1.all_results, f"results_{client_1.base_filename}.csv", ["name", "acc", "total_time", "freeze_degree"])
    client_1.output_csv(client_1.all_metrics, f"metrics_{client_1.base_filename}.csv", ["name", "total_training_time", "total_trainable_weights", "save_transmission_ratio"])
    print(client_1.all_results)
    print(f'Total training time: {datetime.timedelta(seconds= end-start)}')
    client_1.plot_figure()
