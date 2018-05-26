import torch
import torch.optim as optim
import torch.nn.functional as F

from brokkr.benchmarks.log import log_progress
from brokkr.benchmarks.meters import AverageMeter

from brokkr.benchmarks.mnist import MNISTLoaders
from brokkr.benchmarks.mnist.models.onecnn import Net
from brokkr.benchmarks.mnist.parser import parse_args

import time
import logging
import logging.config


logging.config.fileConfig('../logging.conf', defaults={'logfilename': './logs/main.log'})
logger = logging.getLogger(__name__)


def train(args, model, device, train_loader, optimizer, epoch, meters):
    lossmeter = meters['trainloss']
    batchtime = meters['traintime']

    n_batches = len(train_loader)

    model.train()
    end = time.time()
    #with torch.autograd.profiler.emit_nvtx():
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        lossmeter.update(loss.item())
        batchtime.update(time.time() - end)
        end = time.time()

        if batch_idx % args.log_interval == 0:
            log_progress('Train', epoch, args.epochs, batch_idx, n_batches, batchtime, lossmeter)

    lossmeter.reset()
    batchtime.reset()


def test(args, model, device, test_loader, epoch, meters):
    lossmeter = meters['testloss']
    batchtime = meters['testtime']

    n_samples = len(test_loader.dataset)
    n_batches = len(test_loader)

    model.eval()
    test_loss = 0
    correct = 0
    end = time.time()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            test_loss += loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            lossmeter.update(loss)
            batchtime.update(time.time() - end)
            end = time.time()
            accuracy = 100. * correct / n_samples

            if batch_idx % args.log_interval == 0:
                log_progress('Test', epoch, args.epochs, batch_idx, n_batches, batchtime, lossmeter)

    lossmeter.reset()
    batchtime.reset()
    print(f'\nEpoch: {epoch}, Test Accuracy: {accuracy}\n')


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    dataloaders = MNISTLoaders(batch_size=64)
    train_loader = dataloaders.train_loader(args.data, kwargs)
    test_loader = dataloaders.test_loader(args.data, kwargs)

    train_meters = {
      'trainloss': AverageMeter(name='trainloss'),
      'traintime': AverageMeter(name='traintime'),
    }

    test_meters = {
      'testloss': AverageMeter(name='testloss'),
      'testtime': AverageMeter(name='testtime'),
    }

    runtime = AverageMeter(name='runtime')

    end = time.time()
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, train_meters)
        test(args, model, device, test_loader, epoch, test_meters)
        runtime.update(time.time() - end)
        end = time.time()

    print(f"Job's done! Total runtime: {runtime.sum} seconds, Average epoch runtime: {runtime.avg} seconds")


if __name__=="__main__":
    main()
