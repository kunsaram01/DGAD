from config.dataset_config import getData
from eval_utils import test_autoattack, test_robust
from networks.mobilenetv2 import MobileNetV2
from networks.wideresnet import WideResNet, Yao_WideResNet
from networks.resnet import ResNet18, ResNet34
from networks.preresnet import PreActResNet18
from networks.vgg import VGG
import torch.backends.cudnn as cudnn
import argparse
import torch
import torch.nn as nn
from collections import OrderedDict

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', type=str, default='Tiny_Image')
parser.add_argument('--model', type=str, default='resnet18')
parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--method', type=str, default='Plain_Madry')
parser.add_argument('--teacher_model', type=str, default='teacher_model')
parser.add_argument('--bs', default=200, type=int)
parser.add_argument('--eps', default=8/255, type=float)
parser.add_argument('--steps', default=10, type=int)
parser.add_argument('--random-start', default=1, type=int)
parser.add_argument('--coeff', default=0.1, type=float)  # for jsma, cw, ela
args = parser.parse_args()


num_classes, train_data, test_data = getData(args.dataset)

trainloader = torch.utils.data.DataLoader(
    train_data,
    batch_size=args.bs,
    shuffle=True,
    num_workers=4,
    pin_memory=True)
testloader = torch.utils.data.DataLoader(
    test_data,
    batch_size=args.bs,
    shuffle=False,
    num_workers=4,
    pin_memory=True)


# Model
if args.model == 'mobilenetV2':
    net = MobileNetV2(num_classes=num_classes)
elif args.model == 'resnet18':
    net = ResNet18(num_classes)
elif args.model == 'presnet18':
    net = PreActResNet18()
elif args.model == 'resnet34':
    net = ResNet34(num_classes)
elif args.model == 'vgg16':
    net = VGG('VGG16')
elif args.model == 'wideresnet34_10':
    net = WideResNet(num_classes=num_classes)
elif args.model == 'Chen2021WRN34_10':
    net = Yao_WideResNet(num_classes=num_classes, depth=34, widen_factor=10, sub_block1=True)
elif args.model == 'wideresnet34_20':
    net = Yao_WideResNet(num_classes=num_classes, depth=34, widen_factor=20, sub_block1=False)
else:
    raise NotImplementedError


use_cuda = torch.cuda.is_available()
print('use_cuda:%s' % str(use_cuda))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    cudnn.benchmark = True

model_path = args.model_path  # The model path you want to evaluate
print('model path:', model_path)

if args.model == 'wideresnet34_20':
    teacher_state_dict = torch.load(model_path)
    state_dict = OrderedDict()
    for k in list(teacher_state_dict.keys()):
        state_dict[k[7:]] = teacher_state_dict.pop(k)
    teacher_state_dict = state_dict
    net.load_state_dict(teacher_state_dict)
elif args.model == 'Chen2021WRN34_10':
    teacher_state_dict = torch.load(model_path)
    net.load_state_dict(teacher_state_dict)
else:
    net.load_state_dict(torch.load(model_path)['net'])

net.to(device)
net.eval()

print(args.model)
if args.method not in ['Plain_Madry', 'TRADES']:
    print(args.teacher_model)

print('Evaluate FGSM:')
test_robust(net, attack_type='fgsm', c=args.eps, num_classes=num_classes,
            testloader=testloader, loss_fn=nn.CrossEntropyLoss(), req_count=10000)
print('Evaluate PGD:')
test_robust(net, attack_type='pgd', c=args.eps, num_classes=num_classes,
            testloader=testloader, loss_fn=nn.CrossEntropyLoss(), req_count=10000)
print('Evaluate CW:')
test_robust(net, attack_type='cw', c=args.coeff, num_classes=num_classes,
            testloader=testloader, loss_fn=nn.CrossEntropyLoss(), req_count=10000)
print('Evaluate AA:')
test_autoattack(net, testloader, norm='Linf', eps=args.eps,
                version='standard', verbose=False)



