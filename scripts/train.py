from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from plr_exercise.model.cnn import Net
import wandb
import optuna

wandb.login()


def train(args, model, device, train_loader, optimizer, epoch, trial=None):
    """Trains the model for a single epoch.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        model (torch.nn.Module): The PyTorch model to train.
        device (torch.device): The device (CPU or GPU) to use for training.
        train_loader (torch.utils.data.DataLoader): Dataloader for training data.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        epoch (int): The current epoch number.
        trial (optuna.trial.Trial, optional): An Optuna trial object, if used for hyperparameter tuning.
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            # WandB – log the current value of the training loss
            wandb.log({"training_loss": loss.item()})
            if trial is not None:
                trial.report(loss.item(), epoch)  # Report intermediate values to Optuna
            if args.dry_run:
                break


def test(model, device, test_loader, epoch):
    """Evaluates the model on the test set.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        device (torch.device): The device (CPU or GPU) to use for evaluation.
        test_loader (torch.utils.data.DataLoader): Dataloader for test data.
        epoch (int): The current epoch number (for logging purposes).

    Returns:
        float: The accuracy on the test set.
    """
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:

            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    # WandB – log the current value of the training loss
    wandb.log({"test_loss": test_loss})

    accuracy = 100.0 * correct / len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), accuracy
        )
    )
    return accuracy  # Return the accuracy for Optuna


def main():
    """Parses command-line arguments, initializes logging, loads data, performs
    hyperparameter optimization with Optuna, and trains the model with the best hyperparameters.
    """
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=1000, metavar="N", help="input batch size for testing (default: 1000)"
    )
    parser.add_argument("--epochs", type=int, default=2, metavar="N", help="number of epochs to train (default: 14)")
    parser.add_argument("--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)")
    parser.add_argument("--gamma", type=float, default=0.7, metavar="M", help="Learning rate step gamma (default: 0.7)")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--dry-run", action="store_true", default=False, help="quickly check a single pass")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument("--save-model", action="store_true", default=False, help="For Saving the current Model")
    args = parser.parse_args()

    # WandB – Initialize a new run
    run = wandb.init(
        project="plr-exercise",
        dir="../results/",
        config={
            "learning_rate": parser.parse_args().lr,
            "architecture": "CNN",
            "dataset": "MNIST",
            "epochs": parser.parse_args().epochs,
            "batch_size": args.batch_size,
        },
    )
    # WandB – log code as artifact
    code_artifact = wandb.Artifact("project-code", type="code")
    code_artifact.add_dir("./")
    run.log_artifact(code_artifact)
    artifact = run.use_artifact("gtonetti/plr-exercise/project-code:latest", type="code")
    artifact_dir = artifact.download("../results/artifact/")

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # Optuna
    def objective(trial):
        """Defines the objective function for Optuna's hyperparameter search.

        Args:
            trial (optuna.trial.Trial): An Optuna trial object.

        Returns:
            float: The final test accuracy achieved with the suggested hyperparameters.
        """
        # Hyperparamters to be tuned by Optuna
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        momentum = trial.suggest_float("momentum", 0.0, 0.99)
        gamma = trial.suggest_float("gamma", 0.1, 0.99)

        # Integrate with existing ArgumentParser
        args.lr = lr
        args.batch_size = batch_size
        args.momentum = momentum
        args.gamma = gamma

        # Create model and optimizer with Optuna's values
        model = Net().to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

        # Training and Evaluation Loop
        for epoch in range(args.epochs):
            train(args, model, device, train_loader, optimizer, epoch, trial)
            test(model, device, test_loader, epoch)
            scheduler.step()  # Assuming you have a learning rate scheduler

        # Fetch and return the final accuracy
        accuracy = test(model, device, test_loader, epoch)

        return accuracy

    study = optuna.create_study(direction="maximize")  # Maximize test accuracy
    study.optimize(objective, n_trials=10)

    best_params = study.best_params
    print("Best params:", best_params)
    args.lr = best_params["lr"]
    args.batch_size = best_params["batch_size"]
    args.momentum = best_params["momentum"]
    args.gamma = best_params["gamma"]
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(args.epochs):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, epoch)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
