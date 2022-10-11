import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max -
                     lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


class BaseTrainer:
    def __init__(self,
                 net: nn.Module,
                 args: None,
                 train_loader: DataLoader,
                 processed_text_inputs: None,
                 learning_rate: float = 0.1,
                 momentum: float = 0.9,
                 weight_decay: float = 0.0005,
                 epochs: int = 100) -> None:
        assert processed_text_inputs != None and args != None
        self.args = args
        self.net = net
        self.train_loader = train_loader
        self.processed_text_inputs = processed_text_inputs.to(device)

        self.optimizer = torch.optim.SGD(
            net.parameters(),
            learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                epochs * len(train_loader),
                1,  # since lr_lambda computes multiplicative factor
                1e-6 / learning_rate,
            ),
        )

    def train_epoch(self):
        if self.args.model_type == 'clip':
            self.net.eval()  # enter train mode
        else:
            self.net.train()  # enter train mode

        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)

        for train_step in tqdm(range(1, len(train_dataiter) + 1)):
            # for train_step in tqdm(range(1, 5)):
            batch = next(train_dataiter)
            cur_input = {}
            # data = batch['data'].cuda()
            cur_input['pixel_values'] = batch['data'].cuda()
            cur_input['input_ids'] = self.processed_text_inputs['input_ids']
            cur_input['attention_mask'] = self.processed_text_inputs['attention_mask']
            target = batch['soft_label'].cuda()
            # forward
            # logits = self.net(data)
            outputs = self.net(**cur_input)
            logits = outputs.logits_per_image
            # print("logits.size(): ", logits.size())
            # logits.size(): [batch_size, 56]
            # target.size(): [batch_size, 56]
            loss = F.binary_cross_entropy_with_logits(logits,
                                                      target,
                                                      reduction='sum')
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        metrics = {}
        metrics['train_loss'] = loss_avg

        return metrics
