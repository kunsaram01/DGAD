import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


def adaad_inner_loss(model,
                     teacher_model,
                     x_natural,
                     step_size=2 / 255,
                     steps=10,
                     epsilon=8 / 255,
                     BN_eval=True,
                     random_init=True,
                     clip_min=0.0,
                     clip_max=1.0):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='none')
    if BN_eval:
        model.eval()

    # set eval mode for teacher model
    teacher_model.eval()
    # generate adversarial example
    if random_init:
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    else:
        x_adv = x_natural.detach()
    for _ in range(steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                   F.softmax(teacher_model(x_adv), dim=1))
            loss_kl = torch.sum(loss_kl)
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural -
                                    epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, clip_min, clip_max)

    if BN_eval:
        model.train()
    model.train()

    x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max),
                     requires_grad=False)
    return x_adv


def mtad_inner_loss(model,
                    teacher_model,
                    x_natural,
                    targets,
                    step_size=2 / 255,
                    steps=10,
                    epsilon=8 / 255,
                    BN_eval=True,
                    random_init=True,
                    clip_min=0.0,
                    clip_max=1.0):
    # define CE-loss
    criterion_kl = nn.KLDivLoss(reduction='none')

    if BN_eval:
        model.eval()
    # generate adversarial example
    if random_init:
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    else:
        x_adv = x_natural.detach()
    for _ in range(steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                   F.softmax(teacher_model(x_adv), dim=1))
            loss_kl = torch.sum(loss_kl)
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural -
                                    epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, clip_min, clip_max)

    # test if teacher_net can predict correct labels
    # t_pred_correct_nat = torch.argmax(teacher_model(x_natural), dim=1) == targets
    # t_pred_correct_adv = torch.argmax(teacher_model(x_adv), dim=1) == targets
    #
    # print(torch.where(t_pred_correct_nat == False))

    if BN_eval:
        model.train()
    model.train()

    x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max),
                     requires_grad=False)
    return x_adv


def Madry_PGD(model, x_ori, y,
              step_size=2 / 255,
              steps=10,
              epsilon=8 / 255,
              norm='L_inf',
              BN_eval=True,
              random_init=True,
              clip_min=0.0,
              clip_max=1.0):
    criterion = nn.CrossEntropyLoss()

    if BN_eval:
        model.eval()
    if random_init:
        x_adv = x_ori.detach() + 0.001 * torch.randn(x_ori.shape).cuda().detach()
    else:
        x_adv = x_ori.detach()
    x_adv = torch.clamp(x_adv, clip_min, clip_max)
    if norm == 'L_inf':
        for _ in range(steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_i = criterion(model(x_adv), y, size_average=False)

            grad = torch.autograd.grad(loss_i, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(
                torch.max(x_adv, x_ori - epsilon), x_ori + epsilon)
            x_adv = torch.clamp(x_adv, clip_min, clip_max)
    else:
        raise NotImplementedError
    if BN_eval:
        model.train()
    return x_adv


def AttackPGD(model, x_ori, y,
              step_size=2 / 255,
              steps=10,
              epsilon=8 / 255,
              norm='L_inf',
              BN_eval=True,
              random_init=True,
              clip_min=0.0,
              clip_max=1.0):
    criterion = nn.CrossEntropyLoss(reduction='sum')

    if BN_eval:
        model.eval()
    if random_init:
        x_adv = x_ori.detach() + torch.zeros_like(x_ori).uniform_(-epsilon, epsilon)
    else:
        x_adv = x_ori.detach()
    # x_adv = torch.clamp(x_adv, clip_min, clip_max)
    if norm == 'L_inf':
        for _ in range(steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_i = criterion(model(x_adv), y)

            grad = torch.autograd.grad(loss_i, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(
                torch.max(x_adv, x_ori - epsilon), x_ori + epsilon)
            x_adv = torch.clamp(x_adv, clip_min, clip_max)
    else:
        raise NotImplementedError
    if BN_eval:
        model.train()
    return x_adv


def trades_loss(model,
                x_natural,
                y,
                step_size=0.003,
                epsilon=0.031,
                steps=10,
                distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    for _ in range(steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                   F.softmax(model(x_natural), dim=1))
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    # model.train()
    #
    # x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # # zero gradient
    # optimizer.zero_grad()
    # # calculate robust loss
    # logits = model(x_natural)
    # loss_natural = F.cross_entropy(logits, y)
    # loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
    #                                                 F.softmax(model(x_natural), dim=1))
    # loss = loss_natural + beta * loss_robust
    return x_adv


def rslad_inner_loss(model,
                     teacher_model,
                     x_natural,
                     targets,
                     BN_eval=True,
                     random_init=True,
                     step_size=0.003,
                     epsilon=0.031,
                     steps=10):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False, reduce=False)
    if BN_eval:
        model.eval()

    teacher_model.eval()

    # batch_size = len(x_natural)
    # generate adversarial example
    if random_init:
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    else:
        x_adv = x_natural.detach()

    for _ in range(steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                   F.softmax(teacher_model(x_natural), dim=1))
            loss_kl = torch.sum(loss_kl)
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    if BN_eval:
        model.train()
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    # model.train()
    #
    # x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # # zero gradient
    # optimizer.zero_grad()
    # logits = model(x_adv)
    return x_adv
