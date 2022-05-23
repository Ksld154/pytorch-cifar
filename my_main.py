'''Train CIFAR10 with PyTorch.'''
from calendar import EPOCH
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchsummary

import torchvision
import torchvision.transforms as transforms

import torchsummary 
import os
import argparse

from models import *
from utils import progress_bar


EPOCHS = 50
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


def opts_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--epochs', default=50, type=int, help='Train epochs')
    parser.add_argument('--gpu', default=0, type=int, help='GPU ID')


    args = parser.parse_args()
    return args

# Data
def data_preprossing():
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader

# Model
class Model():
    def __init__(self, args, trainloader, testloader, freeze_degree) -> None:
        self.args = args
        self.trainloader = trainloader
        self.testloader = testloader
        
        self.freeze_degree = freeze_degree
        self.accuracy = []

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net, self.criterion, self.optimizer, self.scheduler = self.build_model()

    
    def build_model(self):
        print('==> Building model..')
        # net = VGG('VGG19')
        # net = ResNet18()
        # net = PreActResNet18()
        # net = GoogLeNet()
        # net = DenseNet121()
        # net = ResNeXt29_2x64d()
        # net = MobileNet()
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

        # torchsummary.summary(net, (3,28,28))
        # print(net.__dict__)
        # print(net.modules)

        # print(net.layers)
        # print(len(net.layers))
        # net.layers[2][1].requires_grad_(False)
        
        # for idx, l in enumerate(net.layers):
        #     print(l)
        #     # if idx <= self.freeze_degree:
        #     #     l.requires_grad_(False)
        #     # print(l.requires_grad_)
        #     # print(len(l.parameters))
        #     print('==============')

        # update optimizer after freezing
        # optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        torchsummary.summary(net, (3,28,28))


        if self.args.resume:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/ckpt.pth')
            self.net.load_state_dict(checkpoint['net'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
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


    def test_epoch(self, epoch):
        global best_acc
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
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': self.net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = acc


    def static_freeze_model(self):
        for idx, l in enumerate(self.net.layers):
            print(l)
            if idx <= self.freeze_degree:
                l.requires_grad_(False)
            print('==============')
        # optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        torchsummary.summary(self.net, (3,28,28))
        


def main():
    args = opts_parser()
    print(f'GPU ID: {args.gpu}')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) 

    trainloader, testloader = data_preprossing()
    primary_trainer = Model(args=args, trainloader=trainloader, testloader=testloader, freeze_degree=0)


    for epoch in range(start_epoch, start_epoch + args.epochs):
        primary_trainer.train_epoch(epoch)
        primary_trainer.test_epoch(epoch)
        primary_trainer.scheduler.step()

        # if epoch == 5:
        #     primary_trainer.static_freeze_model()

    print(primary_trainer.accuracy)
                
            


if __name__ == '__main__':
    main()