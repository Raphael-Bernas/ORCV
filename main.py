import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets

from model_factory import ModelFactory
from train import train_FLIP, train_ADV
import wandb


def opts() -> argparse.ArgumentParser:
    """Option Handling Function."""
    parser = argparse.ArgumentParser(description="RecVis A3 training script")
    parser.add_argument(
        "--data",
        type=str,
        default="data_sketches",
        metavar="D",
        help="folder where data is located. train_images/ and val_images/ need to be found in the folder",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="basic_cnn",
        metavar="MOD",
        help="Name of the model for model and transform instantiation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="B",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="experiment",
        metavar="E",
        help="folder where experiment outputs are located.",
    )
    parser.add_argument(
        "--training_method",
        type=str,
        default="basic",
        metavar="TM",
        help="training method to use",
    )
    parser.add_argument(
        "--lamda",
        type=float,
        default=1e-2,
        metavar="La",
        help="regularization parameter for FLIP (default: 1e-2)",
    )
    parser.add_argument(
        "--train_method_iteration",
        type=int,
        default=5,
        metavar="TMI",
        help="number of iterations for the FLIP training method (default: 5)",
    )
    parser.add_argument(
        "--attack_method",
        type=str,
        default="Gaussian",
        metavar="AM",
        help="attack method to use",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-2,
        metavar="Eps",
        help="attack epsilon for the adversarial training method (default: 1e-2)",
    )
    parser.add_argument(
        "--load_model",
        type=str,
        default="None",
        metavar="LM",
        help="Pre-trained model to load. You must not name your model 'None', usually it is of the form model_X.pth",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        metavar="NW",
        help="number of workers for data loading",
    )
    args = parser.parse_args()
    return args


def train(
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
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
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


def validation(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    use_cuda: bool,
) -> float:
    """Default Validation Loop.

    Args:
        model (nn.Module): Model to train
        val_loader (torch.utils.data.DataLoader): Validation data loader
        use_cuda (bool): Whether to use cuda or not

    Returns:
        float: Validation loss
    """
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print(
        "\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            validation_loss,
            correct,
            len(val_loader.dataset),
            100.0 * correct / len(val_loader.dataset),
        )
    )
    wandb.log({"Validation Loss": validation_loss, "Validation Accuracy": 100.0 * correct / len(val_loader.dataset)})
    return validation_loss


def main():
    """Default Main Function."""
    # options
    args = opts()

    # Initialize wandb
    wandb.init(project="recvis-a3", config=args)
    config = wandb.config

    # Check if cuda is available
    use_cuda = torch.cuda.is_available()

    # Set the seed (for reproducibility)
    torch.manual_seed(args.seed)

    # Create experiment folder
    if not os.path.isdir(args.experiment):
        os.makedirs(args.experiment)

    # load model and transform
    if args.load_model != "None":
        state_dict = torch.load(args.load_model)
        model, data_transforms = ModelFactory(args.model_name).get_all()
        model.load_state_dict(state_dict)
    else:
        model, data_transforms = ModelFactory(args.model_name).get_all()
    if use_cuda:
        print("Using GPU")
        model.cuda()
    else:
        print("Using CPU")

    # Data initialization and loading
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + "/train_images", transform=data_transforms),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + "/val_images", transform=data_transforms),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Setup optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Loop over the epochs
    best_val_loss = 1e8
    if args.training_method == "basic":
        print("Training with basic method")
    elif args.training_method == "FLIP":
        print("Training with FLIP method")
    elif args.training_method == "ADV":
        print("Training with adversarial method")
    else:
        raise ValueError("Unknown training method")
    
    for epoch in range(1, args.epochs + 1):
        # training loop
        if args.training_method == "basic":
            train(model, optimizer, train_loader, use_cuda, epoch, args)
        elif args.training_method == "FLIP":
            train_FLIP(model, optimizer, train_loader, use_cuda, epoch, args)
        elif args.training_method == "ADV":
            train_ADV(model, optimizer, train_loader, use_cuda, epoch, args)
        else:
            raise ValueError("Unknown training method")
        # validation loop
        val_loss = validation(model, val_loader, use_cuda)
        if val_loss < best_val_loss:
            # save the best model for validation
            best_val_loss = val_loss
            best_model_file = args.experiment + "/model_best.pth"
            torch.save(model.state_dict(), best_model_file)
            if args.training_method == "FLIP":
                args.lamda = min(args.lamda + 5e-4, 4e-1)
        else:
            if args.training_method == "FLIP":
                args.lamda = max(args.lamda - 5e-4, 1e-4)
        # also save the model every epoch
        model_file = args.experiment + "/model_" + str(epoch) + ".pth"
        torch.save(model.state_dict(), model_file)
        print(
            "Saved model to "
            + model_file
            + f". You can run `python evaluate.py --model_name {args.model_name} --model "
            + best_model_file
            + "` to generate the Kaggle formatted csv file\n"
        )


if __name__ == "__main__":
    main()
