import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets

from model_factory import ModelFactory
from adversarial_update import adversarial_update, lip_constant_estimate
from attacks import fgsm, pgd, gaussian_noise
import wandb


##### This is the training loop for the FLIP method #####
##### FLIP Method: https://github.com/TimRoith/CLIP.git in branch newstruct/ #####
##### More information on FLIP : https://github.com/Raphael-Bernas/Lipschitz_regularization_for_neural_network.git #####

##### FLIP for Finite Lipschitz Regularization is a method to regularize the Lipschitz constant of a neural network. #####

def train_FLIP(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    use_cuda: bool,
    epoch: int,
    args: argparse.ArgumentParser,
) -> None:
    """Default Training Loop.

    Args:
        model (nn.Module): Model to train
        optimizer (torch.optimizer): Optimizer to use
        train_loader (torch.utils.data.DataLoader): Training data loader
        use_cuda (bool): Whether to use cuda or not
        epoch (int): Current epoch
        args (argparse.ArgumentParser): Arguments parsed from command line
    """
    adversarial = lambda x: adversarial_update(model, x, x + 0.05 * torch.randn_like(x), {'name': 'SGD', 'lr': 0.1}, 'sum')
    num_iters = args.train_method_iteration
    lipschitz = lambda u, v: lip_constant_estimate(model, estimation='sum')(u, v)
    lamda = args.lamda
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        data_adv = data.clone().detach().requires_grad_(True)
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = criterion(output, target)
        adv = adversarial(data_adv)
        for _ in range(num_iters):
            adv.step()
        u, v = adv.u, adv.v
        c_reg_loss = lipschitz(u, v)
        loss = loss + lamda * c_reg_loss
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.data.item(),
                )
            )
            wandb.log({"Train Loss": loss.data.item(), "Train Accuracy": 100.0 * correct / len(train_loader.dataset)})

    print(
        "\nTrain set: Accuracy: {}/{} ({:.0f}%)\n".format(
            correct,
            len(train_loader.dataset),
            100.0 * correct / len(train_loader.dataset),
        )
    )

##### This is the training loop for the Adversarial method #####

def train_ADV(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    use_cuda: bool,
    epoch: int,
    args: argparse.ArgumentParser,
) -> None:
    """Default Training Loop.

    Args:
        model (nn.Module): Model to train
        optimizer (torch.optimizer): Optimizer to use
        train_loader (torch.utils.data.DataLoader): Training data loader
        use_cuda (bool): Whether to use cuda or not
        epoch (int): Current epoch
        args (argparse.ArgumentParser): Arguments parsed from command line
    """
    adversarial = args.attack_method
    if adversarial == "FGSM":
        adversarial = fgsm(epsilon = args.epsilon)
    elif adversarial == "PGD":
        adversarial = pgd(epsilon = args.epsilon)
    elif adversarial == "Gaussian":
        adversarial = gaussian_noise(epsilon = args.epsilon)
    else:
        raise ValueError('Unknown adversarial method: ' + str(adversarial))
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        adversarial(model, data, target)
        output = model(data+adversarial.delta.detach())
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.data.item(),
                )
            )
            wandb.log({"Train Loss": loss.data.item(), "Train Accuracy": 100.0 * correct / len(train_loader.dataset)})

    print(
        "\nTrain set: Accuracy: {}/{} ({:.0f}%)\n".format(
            correct,
            len(train_loader.dataset),
            100.0 * correct / len(train_loader.dataset),
        )
    )
