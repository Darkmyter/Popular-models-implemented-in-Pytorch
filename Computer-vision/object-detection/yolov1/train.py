import click
import statistics
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from pathlib import Path
from dataset import get_VOCDataset
from utils import (
    save_checkpoint,
    load_checkpoint,
)
from loss import YoloLoss



class WrappedChainScheduler(torch.optim.lr_scheduler.ChainedScheduler):
    """For some reason, PyTorch's ChainedScheduler class does not allow you to pass keyword arguments
    to the schedulers in the list, preventing you from using ReduceLROnPlateau in a chain. This is a 
    simple wrapper class which fixes that issue.
    """

    def __init__(self, schedulers):
        super().__init__(schedulers)
    
    def step(self, metrics = None):
        for scheduler in self._schedulers:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(metrics)
            else:
                scheduler.step()
        self._last_lr = [group['lr'] for group in self._schedulers[-1].optimizer.param_groups]


def train(device, model, optimizer, loss_function, train_loader, test_loader, scheduler, params, model_name):
    for epoch in range(params['epochs']):
        loop = tqdm(train_loader, leave=True)

        losses = []
        for batch_idx, (x, y) in enumerate(loop):
            x, y = x.to(device), y.to(device)
            loss = loss_function(model(x), y)

            # run autograd
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update progress bar
            losses.append(loss.item())
            loop.set_postfix(loss=statistics.mean(losses), epoch=epoch, lr=scheduler.get_last_lr()[0])
        
        scheduler.step(metrics = statistics.mean(losses))

        if epoch % params['num_epochs_between_checkpoints'] == 0 and epoch:
            save_checkpoint(model, optimizer, scheduler, model_name)

        if params["test"] and epoch % params['num_epochs_between_tests'] == 0:
            pass
            # TODO

    save_checkpoint(model, optimizer, scheduler, model_name)

def init(device, model_name, params):

    model = Yolov1(in_channels=3, split_size=7, num_boxes=2, num_classes=20).to(device)

    optimizer = torch.optim.SGD(
            model.parameters(), 
            momentum = params['momentum'],
            weight_decay = params['weight_decay'], 
            lr = params['lr']
            )
    
    schedular = WrappedChainScheduler([
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor = 0.1, end_factor = 1.0, total_iters = 10),
        torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 105, 135])                            
    ]
    )

    loss_function = YoloLoss().to(device)

    train_dataset = get_VOCDataset(
        csv_file=params['training_csv'],
        img_dir=params['img_dir'],
        label_dir=params['label_dir'],
        augment=True
        )
    
    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = params['batch_size'],
        num_workers = params['num_workers'],
        pin_memory = params['pin_memory'],
        shuffle = True,
        drop_last = True
    )

    test_dataset = get_VOCDataset(
        csv_file=params['test_csv'],
        img_dir=params['img_dir'],
        label_dir=params['label_dir'],
        augment=True
        )
    
    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size = params['batch_size'],
        num_workers = params['num_workers'],
        pin_memory = params['pin_memory'],
        shuffle = True,
        drop_last = True
    )

    return model, optimizer, loss_function, train_loader, test_loader, schedular

@click.command()
@click.option('-m', '--model_name', 'model_name', type=str, prompt = 'Enter model name', help = 'The model to train.')
@click.option('-n', '--new', is_flag=True, help='Generate new model rather than loading from checkpoint')
@click.option('-s', '--seed', type=int, help='Manual seed')
def main(model_name, new, seed):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not seed is None: torch.manual_seed(seed)

    params = {
        "lr": 1e-3,
        "batch_size": 16,
        "weight_decay": 5e-4,
        "momentum": 0.9,
        "epochs": 90,
        "num_workers": 8,
        "pin_memory": True,
        "img_dir": Path("../../../data/voc/images"),
        "label_dir": Path("../../../data/voc/labels"),
        "training_csv": Path("../../../data/voc/train.csv"),
        "test_csv": Path("../../../data/voc/test.csv"),
        "num_epochs_between_checkpoints": 5,
        "num_epochs_between_tests": 10,
        "test": True
    }

    model, optimizer, loss_function, train_loader, test_loader, scheduler = init(device, model_name, params)
    
    if not new: load_checkpoint(model, optimizer, scheduler, model_name)

    train(device, model, optimizer, loss_function, train_loader, test_loader, scheduler, params, model_name)

if __name__ == "__main__":
    main()


