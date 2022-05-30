
import time
import datetime
import copy




import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchsummary

import utils
from constants import *
from model import MyModel
from tools.utils import moving_average


class Trainer():
    def __init__(self, model, freeze_idx, old_obj) -> None:
        self.model = model
        self.args = model.args
        self.name = model.name
        self.model_type = model.name
        
        self.freeze_idx = freeze_idx
        
        if self.model_type == 'mobilenet':
            self.freeze_step_size = MOBILENETV2_FREEZE_STEP
        elif self.model_type == 'resnet':
            self.freeze_step_size = RESNET_FREEZE_STEP
        else:
            self.freeze_step_size = LENET_FREEZE_STEP


        self.total_training_time = datetime.timedelta(seconds=0)
        self.loss = []
        self.loss_delta = []
        self.accuracy = []
        self.cur_layer_weight = []
        self.layer_history = []

        self.utility = []
        self.total_trainable_weights = 0


        for idx, l in enumerate(self.model.net.layers):
            if idx < self.freeze_idx:
                l.requires_grad_(False)
            # print(idx, l)
        # self.model.summary()

        # self.optimizer = optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
        # self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.net.parameters()), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.net.parameters()), lr=self.args.lr, momentum=0.5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        # torchsummary.summary(self.model.net, (3,28,28))

        if old_obj:
            self.loss = old_obj.loss
            self.loss_delta = old_obj.loss_delta
            self.accuracy = old_obj.accuracy
            self.cur_layer_weight = old_obj.cur_layer_weight
            self.utility = old_obj.utility
            self.total_training_time = old_obj.total_training_time
            self.total_trainable_weights = old_obj.total_trainable_weights
            self.layer_history = old_obj.layer_history


    def train_and_test_epoch(self, e):
        # train each batch
        epoch_start = time.time()
        self.model.train_epoch(e)
        epoch_end = time.time()
        
        elapsed_time = datetime.timedelta(seconds= epoch_end-epoch_start)
        self.total_training_time += elapsed_time
        print(f'[Model {self.name}] Epoch training time: {elapsed_time}')

        loss, acc = self.model.test_epoch(e)
        print(loss, acc)
        self.model.scheduler.step()

        self.save_loss_delta(loss)
        self.loss.append(loss)
        self.accuracy.append(acc)
        self.layer_history.append(self.freeze_idx)

        return loss, acc
    

    def save_loss_delta(self, cur_loss):
        if not self.loss:
            self.loss_delta.append(None)
            return

        if self.loss[-1] != None:
            loss_delta = cur_loss - self.loss[-1]
        else:
            loss_delta = None
        self.loss_delta.append(loss_delta)

    def get_model(self):
        return self.model
    
    def calc_frozen_ratio(self):
        # total_weights_cnt = self.net.count_params()
        total_params = sum(p.numel() for p in self.model.net.parameters())
        frozen_params = sum(p.numel() for p in self.model.net.parameters() if not p.requires_grad) + 1 
        trainable_params = total_params - frozen_params
        
        frozen_ratio = (frozen_params) / total_params
        self.total_trainable_weights += trainable_params
        print(f'[Model {self.name}] Total parameters: {total_params}, Frozen parameters: {frozen_params}')

        return frozen_params, frozen_ratio
    
    # deprecated
    def get_model_utility(self, frozen_layers):
        cur_loss = self.loss[-1]
        model_loss_satisfied = True if cur_loss <= LOSS_THRESHOLD * LOSS_THRESHOLD_ALPHA else False
        print(model_loss_satisfied) 
        
        model_utility = 0
        if not model_loss_satisfied:
            # model_utility = (cur_loss-LOSS_THRESHOLD * LOSS_THRESHOLD_ALPHA) / frozen_params_ratio
            if frozen_layers != 0:
                model_utility = (cur_loss-LOSS_THRESHOLD * LOSS_THRESHOLD_ALPHA) ** (frozen_layers*MAGIC_ALPHA)
            else: 
                model_utility = (cur_loss-LOSS_THRESHOLD * LOSS_THRESHOLD_ALPHA)
        else:
            model_utility = 1e-7 / frozen_layers

        return model_utility

    def is_converged(self, pre_epochs=3, window_size=MOVING_AVERAGE_WINDOW_SIZE):
        delta_ma = moving_average(self.loss_delta[pre_epochs:], window_size)
        print(f'Trainer {self.name} loss delta: {delta_ma}')
        if not np.isnan(delta_ma) and delta_ma <= LOSS_CONVERGED_THRESHOLD:
            return True
        else:
            return False

    # deprecated
    def further_freeze(self, pre_epochs, force_freeze):
        # if force_freeze or self.is_converged(pre_epochs):
        #     if self.freeze_idx >= len(self.freeze_options) -1:
        #         return self
            
        #     print(f"Model {self.name} is converge, will advance to next freezing degree{self.freeze_idx+1}")
        #     self.freeze_idx += 1
            
        #     # old_weights = self.get_model().get_weights()
        #     # new_model = keras.models.clone_model(self.get_model())
        #     # new_model.set_weights(old_weights)


        #     new_model = MyModel(args=self.args, trainloader=self.model.trainloader, testloader=self.model.testloader, model_name=self.name)
        #     new_net = copy.deepcopy(self.model.net)
        #     new_model.net = new_net
        #     # new_model.static_freeze_model(freeze_degree=self.freeze_idx+1)

        #     new_trainer = Trainer(model=new_model, freeze_idx=self.freeze_idx, old_obj=self)
        #     new_trainer.set_model_name(f"Frozen_degree_{self.freeze_idx}")
        #     new_trainer.loss_delta.clear()
        
        #     return new_trainer
        
        # return self
        pass

    
    # generate a static freeze trainer with degree=freeze_degree
    def static_freeze(self, freeze_degree):  
        self.freeze_idx = freeze_degree
        print(f"Model {self.name} is set to freezing degree: {freeze_degree}")

        new_model = MyModel(args=self.args, trainloader=self.model.trainloader, testloader=self.model.testloader,  model_name=self.model_type)
        new_net = copy.deepcopy(self.model.net) 
        new_model.net = new_net # copy from current model
        new_model.static_freeze_model(freeze_degree=freeze_degree)

        new_trainer = Trainer(model=new_model, freeze_idx=freeze_degree, old_obj=False)
        new_trainer.set_model_name(f"Frozen_degree_{freeze_degree}")
        new_trainer.loss_delta.clear()
    
        return new_trainer
        

    # generate a secondary trainer object that freeze 1 layer deeper than primary trainer
    def generate_secondary_trainer(self, old_secondary_trainer):
        if self.freeze_idx >= len(self.model.net.layers) -1:
            return copy.deepcopy(self)            
        print(f"Generate a secondary model with freezing degree={self.freeze_idx+1} from primary trainer")
    
        new_model = MyModel(args=self.args, trainloader=self.model.trainloader, testloader=self.model.testloader, model_name=self.model_type)
        new_net = copy.deepcopy(self.model.net)
        new_model.net = new_net
        # new_model.static_freeze_model(freeze_degree=self.freeze_idx+1)

        new_trainer = Trainer(model=new_model, freeze_idx=self.freeze_idx+1, old_obj=old_secondary_trainer)
        new_trainer.name = f"Secondary (Frozen_degree={self.freeze_idx+1})"
        # new_trainer.loss_delta.clear()

        return new_trainer
    
    def set_model_name(self, new_model_name):
        self.name = new_model_name
    
    def static_freeze_test(self, freeze_degree):
        self.model.static_freeze_model(freeze_degree=freeze_degree)