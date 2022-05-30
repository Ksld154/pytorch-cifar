'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn



import torchsummary 
import os

from models import *
from utils import progress_bar


# Model
class MyModel():
    def __init__(self, args, model_name, trainloader, testloader) -> None:
        self.args = args
        self.trainloader = trainloader
        self.testloader = testloader
        self.name = model_name
        
        # self.freeze_degree = freeze_degree  # no use
        self.accuracy = []
        self.best_acc = 0

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net, self.criterion, self.optimizer, self.scheduler = self.build_model(self.name)

    
    def build_model(self, model_name=None):
        print('==> Building model..')
        # net = VGG('VGG19')
        # net = ResNet18()
        # net = PreActResNet18()
        # net = GoogLeNet()
        # net = DenseNet121()
        # net = ResNeXt29_2x64d()
        # net = MobileNet()
        if model_name == 'resnet':
            net = ResNet18()
        elif model_name == 'lenet':
            net = LeNet()
        else:
            net = MobileNetV2()
        # net = DPN92()
        # net = ShuffleNetG2()
        # net = SENet18()
        # net = ShuffleNetV2(1)
        # net = EfficientNetB0()
        # net = RegNetX_200MF()
        # net = SimpleDLA()

        # print(torch.cuda.device_count())
        net = net.to(self.device)

        print(len(net.layers))

        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=self.args.lr, momentum=0.5)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        return net, criterion, optimizer, scheduler


    # Training
    def train_epoch(self, epoch):
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
        train_acc = float(correct/total)
        return train_loss, train_acc

    def test_epoch(self, epoch):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(self.testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # Save checkpoint.
        acc = 100.*correct/total
        self.accuracy.append(acc/100)
        if not self.best_acc or acc > self.best_acc:
            print('Saving..')
            state = {
                'net': self.net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            self.best_acc = acc
        
        test_acc = acc / 100.0
        return test_loss/(batch_idx+1), test_acc

    def static_freeze_model(self, freeze_degree):
        for idx, l in enumerate(self.net.layers):
            # print(l)
            if idx < freeze_degree:
                l.requires_grad_(False)
            # print('==============')

        # self.optimizer = optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
        # self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.args.lr, momentum=0.5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        self.summary()
        # torchsummary.summary(self.net, (3,28,28), device=self.device)
    

    def summary(self):
        if self.name == 'lenet':
            torchsummary.summary(self.net, (3,32,32), device=self.device)
        else:
            torchsummary.summary(self.net, (3,28,28), device=self.device)
