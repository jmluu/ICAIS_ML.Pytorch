from __future__ import print_function, division, absolute_import
import  argparse,time,logging,os
import  random, shutil, warnings

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import models

from utils import *

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
    parser.add_argument('--data', metavar='DIR', default='data/Cifar-10',
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='cifar_resnet20',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=120, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0001, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-period', type=int, default=0,
                        help='period in epoch for learning rate decays. default is 0 (has no effect).')
    parser.add_argument('--lr-decay-epoch', type=str, default='40,60',
                        help='epochs at which learning rate decays. default is 40,60.')
    parser.add_argument('--save-period', type=int, default=10,
                        help='period in epoch of model saving.')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', default=False,
                        help='evaluate model on validation set')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')


    args = parser.parse_args()
    return args


args = parse_args()

# set up train folder
current_time = time.strftime('%m%d-%H-%M-%S-', time.localtime(time.time()))
current = 'checkpoints/' + current_time + args.arch + '/'
create_exp_dir(current)
filehandler = logging.FileHandler(os.path.join(current, 'log.txt'))
streamhandler = logging.StreamHandler()

logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)
logger.info(args)


best_acc1 = 0


def main():
    global best_acc1
    print(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_epoch = args.start_epoch             # start from epoch 0 or last checkpoint epoch
    classes = 10

    lr_decay_epoch = [int(i) for i in args.lr_decay_epoch.split(',')] + [np.inf]

    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=classes)

    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    logger.info(model)

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                                nesterov=True, weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss().to(device)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            strs = ("=> loaded checkpoint '{}' (epoch {}, best_acc1 {})"
                    .format(args.resume, checkpoint['epoch'], best_acc1))

            logger.info(strs)

        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))


    # Data
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

    trainset = torchvision.datasets.CIFAR10(root=args.data, train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    testset = torchvision.datasets.CIFAR10(root=args.data, train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    classes_name = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    lr = args.lr
    lr_decay_count = 0

    for epoch in range(start_epoch, args.epochs):
        if epoch == lr_decay_epoch[lr_decay_count] :
            lr = lr * args.lr_decay
            lr_decay_count += 1
            adjust_learning_rate(optimizer, lr)

        train(trainloader, model, criterion, optimizer, epoch, device)
        acc1 = test(testloader, model, criterion, epoch, device)

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        logger.info('epoch : %d, best acc : %f' % (epoch, best_acc1))
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best, current=current)

# Training
def train(trainloader, model, criterion, optimizer, epoch, device):
    logger.info('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx+1)%30 == 0 :
            log = 'Loss: %.4f | Acc: %.3f%% (%d/%d)'\
                  % (train_loss/(batch_idx+1), 100.*correct/total, correct, total)
            logger.info(log)


def test(testloader, model, criterion, epoch, device):
    global best_acc1
    logger.info('---- Test  --------')
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    final_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % 30 == 0:
                log = ('Loss: %.4f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
                logger.info(log)

            final_loss = test_loss/(batch_idx+1)
    # Save checkpoint.
    acc = 100.*correct/total


    return acc

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, current, filename='checkpoint.pth.tar'):
    checkpoint_filename = current + filename
    torch.save(state, checkpoint_filename)
    if is_best:
        shutil.copyfile(checkpoint_filename, current + 'model_best.pth.tar')





if __name__ == '__main__':
    main()



