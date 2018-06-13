import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
os.environ["CUDA_VISILE_DEVICE"] = "1"
import argparse

from models import *
from utils import progress_bar

parser = argparse.ArgumentParser(description='Pytorch for Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # nn.init.constant_(m.weight.data, 0)
        nn.init.xavier_normal_(m.weight.data)
        # print(m.weight)


print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.RandomCrop(100, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.4717, 0.4717, 0.4717], [0.2189, 0.2189, 0.2189])
])

transform_train2 = transforms.Compose([
    transforms.Resize((100,100)),
    transforms.RandomCrop(100, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.5, 0.5, 0.5, 0.3),
    transforms.RandomAffine(10),
    transforms.ToTensor(),
    transforms.Normalize([0.4755, 0.4755, 0.4755], [0.2103, 0.2103, 0.2103])
    ])

transform_test = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize([0.4293, 0.4099, 0.3822], [0.2469, 0.2528, 0.2525])
])

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
#
# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

data_dir = '/media/disk4/share/DataSet/Driver_Drowsiness/face/'

trainset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train_folder'), transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)

testset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'val_folder'), transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=4)

data_dir2 = '/media/disk4/share/DataSet/Driver_Drowsiness/ypb_eye_dataset/'

trainset2 = torchvision.datasets.ImageFolder(os.path.join(data_dir2, 'dataset_B_FacialImages'), transform=transform_train2)
trainloader2 = torch.utils.data.DataLoader(trainset2, batch_size=32, shuffle=True, num_workers=4)


# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'))
classes = trainset.classes

if args.resume:
    print('==> Resume model..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    # net.load_state_dict(checkpoint['net'])
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    net = MyNet()
    net.apply(weights_init)


net = net.to(device)
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test(epoch):
    global best_acc
    print('Best Acc : {}'.format(best_acc))
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            # 'net': net.state_dict(),
            'net': net if device=='cuda' else net,
            'acc': acc,
            'epoch':epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc



for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)

# from utils import get_mean_and_std
# print(get_mean_and_std(trainset2))