import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.optim import Adam

from dataset import gen_dataloaders
from nets.MobileNetV2_unet import MobileNetV2_unet


# count number of model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(data_loader, model, optimizer, criterion):
    model.train()
    running_loss = 0.0
    count = 0

    for batch_idx, (inputs, labels) in enumerate(data_loader):
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        optimizer.zero_grad()

        # with torch.set_grad_enabled(True):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        count += inputs.size(0)

        if batch_idx % args.log_interval != 0:
            continue

        print('[{}/{} ({:0.0f}%)]\tLoss: {:0.3f}'.format(
            batch_idx * len(inputs),
            len(data_loader.dataset),
            100. * batch_idx / len(data_loader),
            loss.item()))

    epoch_loss = running_loss / count
    print('[End of train epoch]\tLoss: {:0.5f}'.format(epoch_loss))

    return epoch_loss


def test(data_loader, model, criterion):
    model.eval()
    running_loss = 0.0
    count = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        count += inputs.size(0)

    epoch_loss = running_loss / count
    print('[End of test epoch]\tLoss: {:0.5f}'.format(epoch_loss))

    return epoch_loss


def main():
    # Tensorboard writer
    writer = SummaryWriter(args.log_dir)
    save_filename = args.model_dir

    # Data
    train_loader, valid_loader = gen_dataloaders(args.data_folder,
                                                 val_split=0.05, shuffle=True,
                                                 batch_size=args.batch_size,
                                                 seed=args.seed,
                                                 img_size=224,
                                                 cuda=args.cuda
                                                 )

    # Model
    model = MobileNetV2_unet(pre_trained=args.pre_trained).to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    test_losses = []
    best_loss = np.inf
    for epoch in range(1, args.num_epochs + 1):
        print("================== Epoch: {} ==================".format(epoch))

        train_loss = train(train_loader, model, optimizer, criterion)
        val_loss = test(valid_loader, model, criterion)

        # Logs
        writer.add_scalar('loss/train/loss', train_loss, epoch)
        writer.add_scalar('loss/test/loss', val_loss, epoch)

        test_losses.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss

            # Save model
            with open('{0}/best.pt'.format(save_filename), 'wb') as f:
                torch.save(model.state_dict(), f)

        # Early stopping
        if args.patience is not None and epoch > args.patience + 1:
            loss_array = np.array(test_losses)

            if all(loss_array[-args.patience:] - best_loss >
                   args.early_stopping_eps):
                break

    print("Model saved at: {0}/best.pt".format(save_filename))
    print("# Parameters: {}".format(count_parameters(model)))

    return


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Patchy VAE')

    # Dataset
    parser.add_argument('--dataset', type=str, default='lfw',
                        help='name of the dataset (default: lfw)')
    parser.add_argument('--data-folder', type=str, default='./data',
                        help='name of the data folder (default: ./data)')
    parser.add_argument('--workers', type=int, default=1,
                        help='number of image preprocessing (default: 1)')

    # Model
    parser.add_argument('--arch', type=str, default='patchy',
                        help='model architecture (default: patchy)')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=8,
                        help='batch size (default: 8)')
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='number of epochs (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training (default: False)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='learning rate for Adam optimizer (default: 3e-4)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')

    # Early Stopping
    parser.add_argument('--patience', type=int, default=None,
                        help='patience for early stopping (default: None)')
    parser.add_argument('--early-stopping-eps', type=int, default=1e-5,
                        help='patience for early stopping (default: 1e-5)')

    # Miscellaneous
    parser.add_argument('--pre-trained', type=str, default=None,
                        help='path of pre-trained weights (default: None)')
    parser.add_argument('--output-folder', type=str, default='./scratch',
                        help='name of the output folder (default: ./scratch)')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    # Slurm
    if 'SLURM_JOB_NAME' in os.environ and 'SLURM_JOB_ID' in os.environ:
        # running with sbatch and not srun
        if os.environ['SLURM_JOB_NAME'] != 'bash':
            args.output_folder = os.path.join(args.output_folder,
                                              os.environ['SLURM_JOB_ID'])
    else:
        args.output_folder = os.path.join(args.output_folder, str(os.getpid()))

    # Create logs and models folder if they don't exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    log_dir = os.path.join(args.output_folder, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    model_dir = os.path.join(args.output_folder, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    args.log_dir = log_dir
    args.model_dir = model_dir

    main()
