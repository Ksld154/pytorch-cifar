
import time
import datetime
import copy

import utils
from constants import *


import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K

# BATCH_SIZE = 64
# FREEZE_OPTIONS = [0, 2, 4, 6, 7]
# MOVING_AVERAGE_WINDOW_SIZE = 3
# LOSS_COVERGED_THRESHOLD = 0.01

# LOSS_THRESHOLD = 1.2
# LOSS_THRESHOLD_ALPHA = 1.2

class Trainer():
    def __init__(self, model, data, freeze_idx, recompile, old_obj) -> None:
        self.model = model
        self.data = data
        self.recompile = recompile
        self.name = self.model._name
        self.total_training_time = datetime.timedelta(seconds=0)

        self.freeze_idx = freeze_idx
        self.freeze_layers = FREEZE_OPTIONS[freeze_idx]

        self.loss = []
        self.loss_delta = []
        self.accuracy = []
        self.cur_layer_weight = []
        self.layer_history = []

        self.utility = []
        self.total_trainable_weights = 0

        if self.recompile:
            for l in range(self.freeze_layers):
                self.model.layers[l].trainable = False

            sgd = keras.optimizers.SGD(learning_rate=0.01,
                                       momentum=0.0,
                                       decay=1e-4,
                                       nesterov=False)
            self.model.compile(loss=tf.keras.losses.categorical_crossentropy,
                               optimizer=sgd,
                               metrics=['accuracy'])
            # self.model.summary()
        if old_obj:
            self.loss = old_obj.loss
            self.loss_delta = old_obj.loss_delta
            self.accuracy = old_obj.accuracy
            self.cur_layer_weight = old_obj.cur_layer_weight
            self.utility = old_obj.utility
            self.total_training_time = old_obj.total_training_time
            self.total_trainable_weights = old_obj.total_trainable_weights
            self.layer_history = old_obj.layer_history


    def train_epoch(self):
        # train each batch
        epoch_start = time.time()
        for x, y in zip(self.data.x_train_batch, self.data.y_train_batch):
            y_pred = self.model.train_on_batch(x, y)
            # print(y_pred)

        loss, accuracy = self.model.evaluate(self.data.x_test,
                                             self.data.y_test,
                                             batch_size=BATCH_SIZE,
                                             )
        # lr = K.get_value(self.model.optimizer._decayed_lr(tf.float32))
        # print(f"Learning rate: {lr}")
        epoch_end = time.time()
        elapsed_time = datetime.timedelta(seconds= epoch_end-epoch_start)
        self.total_training_time += elapsed_time
        print(f'[Model {self.name}] Epoch training time: {elapsed_time}')

        self.save_loss_delta(loss)
        self.loss.append(loss)
        self.accuracy.append(accuracy)
        self.layer_history.append(self.freeze_idx)
        return loss, accuracy
    
    def train_epoch_fit(self):
        # train each batch
        epoch_start = time.time()
        train_result = self.model.fit(self.data.x_train, self.data.y_train,epochs=1)
        train_loss = train_result.history['loss'][0]
        print(train_loss)
        # print(train_result.history)

        loss, accuracy = self.model.evaluate(self.data.x_test,
                                             self.data.y_test,
                                             batch_size=BATCH_SIZE,
                                             )
        # lr = K.get_value(self.model.optimizer._decayed_lr(tf.float32))
        # print(f"Learning rate: {lr}")
        epoch_end = time.time()
        elapsed_time = datetime.timedelta(seconds= epoch_end-epoch_start)
        self.total_training_time += elapsed_time
        print(f'[Model {self.name}] Epoch training time: {elapsed_time}')

        self.save_loss_delta(loss)
        self.loss.append(loss)
        self.accuracy.append(accuracy)
        return loss, accuracy


    def save_layer_weights(self, current_layer_idx):
        current_layer_weights = self.model.get_weights()[
            FREEZE_OPTIONS[current_layer_idx]]

        layer_sum = np.sum(current_layer_weights **
                           2) / current_layer_weights.size
        layer_sum2 = np.sum(current_layer_weights) / np.sum(
            current_layer_weights**2)
        self.cur_layer_weight.append(layer_sum2)

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
    
    def get_frozen_ratio(self):
        total_weights_cnt = self.model.count_params()

        frozen_weights_cnt = 1
        for _, layer in enumerate(self.model.layers):
            if layer.trainable == False:
                frozen_weights_cnt += layer.count_params()
        frozen_ratio = (frozen_weights_cnt) / total_weights_cnt
        # print(f'[Model {self.name}] Total parameters: {total_weights_cnt}, Frozen parameters: {frozen_weights_cnt}')
        # print(f'[Model {self.name}] Frozen parameter ratio: {frozen_ratio}')
        self.total_trainable_weights += total_weights_cnt-frozen_weights_cnt

        return frozen_weights_cnt, frozen_ratio
    
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

    def is_converged(self, pre_epochs):
        delta_ma = utils.moving_average(self.loss_delta[pre_epochs:], MOVING_AVERAGE_WINDOW_SIZE)
        print(f'Model {self.name} loss delta: {delta_ma}')
        if not np.isnan(delta_ma) and delta_ma <= LOSS_COVERGED_THRESHOLD:
            return True
        else:
            return False

    def further_freeze(self, pre_epochs, force_freeze):
        if force_freeze or self.is_converged(pre_epochs):
            if self.freeze_idx >= len(FREEZE_OPTIONS) -1:
                return self
            
            print(f"Model {self.name} is converge, will advance to next freezing degree{self.freeze_idx+1}")
            self.freeze_idx += 1
            old_weights = self.get_model().get_weights()
            new_model = keras.models.clone_model(self.get_model())
            new_model.set_weights(old_weights)
            new_trainer = Trainer(new_model, self.data, self.freeze_idx, True, self)
            new_trainer.set_model_name(f"Frozen_degree_{self.freeze_idx+1}")
            new_trainer.loss_delta.clear()
        
            return new_trainer
        
        return self
    
    def static_further_freeze(self, freeze_idx):

            
        self.freeze_idx = freeze_idx
        print(f"Model {self.name} is set to freezing degree: {self.freeze_idx}")

        old_weights = self.get_model().get_weights()
        new_model = keras.models.clone_model(self.get_model())
        new_model.set_weights(old_weights)

        new_trainer = Trainer(new_model, self.data, self.freeze_idx, True, self)
        new_trainer.set_model_name(f"Frozen_degree_{self.freeze_idx}")
        new_trainer.loss_delta.clear()
    
        return new_trainer
        


    # generate a secondary trainer object that freeze 1 layer deeper than primary trainer
    def generate_further_freeze_trainer(self, old_secondary_trainer):
        if self.freeze_idx >= len(FREEZE_OPTIONS) -1:
            return copy.deepcopy(self)
            
        print(f"Generate a secondary model with freezing degree{self.freeze_idx+1} from primary trainer")
        new_freeze_idx = self.freeze_idx+1
        old_weights = self.get_model().get_weights()
        new_model = keras.models.clone_model(self.get_model())
        new_model.set_weights(old_weights)
        
        new_trainer = Trainer(new_model, self.data, new_freeze_idx, True, old_secondary_trainer)
        new_trainer.set_model_name(f"Frozen_degree_{new_freeze_idx}")
        # new_trainer.loss_delta.clear()
    
        return new_trainer
    
    def set_model_name(self, new_model_name):
        self.model._name = new_model_name
        self.name = new_model_name