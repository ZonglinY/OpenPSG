import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time

import torch
from transformers import CLIPProcessor, CLIPModel
from dataset import PSGClsDataset
from evaluator import Evaluator
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from trainer import BaseTrainer
from utils import relation_texts

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='res50')
parser.add_argument('--epoch', type=int, default=10)
# parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr', type=float, default=3e-9)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--model_type', type=str, default="clip")
args = parser.parse_args()

if not (args.model_type == 'clip' or args.model_type == 'resnet'):
    raise NotImplementedError

savename = f'{args.model_name}_e{args.epoch}_lr{args.lr}_bs{args.batch_size}_m{args.momentum}_wd{args.weight_decay}'
os.makedirs('./checkpoints', exist_ok=True)
os.makedirs('./results', exist_ok=True)

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# loading image dataset
train_dataset = PSGClsDataset(stage='train', clip_processor=processor, num_classes=56)
train_dataloader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=8)

val_dataset = PSGClsDataset(stage='val', clip_processor=processor, num_classes=56)
val_dataloader = DataLoader(val_dataset,
                            batch_size=32,
                            shuffle=False,
                            num_workers=8)

test_dataset = PSGClsDataset(stage='test', clip_processor=processor, num_classes=56)
test_dataloader = DataLoader(test_dataset,
                             batch_size=32,
                             shuffle=False,
                             num_workers=8)
# loading text dataset
processed_text_inputs = processor(text=relation_texts, return_tensors="pt", padding=True)
print("processed_text_inputs.size(): ", processed_text_inputs['input_ids'].size())
print('Data Loaded...', flush=True)

# loading model
# model = resnet50(pretrained=True)
# model.fc = torch.nn.Linear(2048, 56)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model.cuda()
print('Model Loaded...', flush=True)

# loading trainer
trainer = BaseTrainer(model,
                      args,
                      train_dataloader,
                      processed_text_inputs,
                      learning_rate=args.lr,
                      momentum=args.momentum,
                      weight_decay=args.weight_decay,
                      epochs=args.epoch)
evaluator = Evaluator(model, processed_text_inputs, k=3)

# train!
print('Start Training...', flush=True)
begin_epoch = time.time()
best_val_recall = 0.0
for epoch in range(0, args.epoch):
    train_metrics = trainer.train_epoch()
    val_metrics = evaluator.eval_recall(val_dataloader)

    # show log
    print(
        '{} | Epoch {:3d} | Time {:5d}s | Train Loss {:.4f} | Test Loss {:.3f} | mR {:.2f}'
        .format(savename, (epoch + 1), int(time.time() - begin_epoch),
                train_metrics['train_loss'], val_metrics['test_loss'],
                100.0 * val_metrics['mean_recall']),
        flush=True)

    # save model
    if val_metrics['mean_recall'] >= best_val_recall:
        torch.save(model.state_dict(), f'./checkpoints/{savename}_best.ckpt')
        best_val_recall = val_metrics['mean_recall']

print('Training Completed...', flush=True)

# saving result!
print('Loading Best Ckpt...', flush=True)
checkpoint = torch.load(f'checkpoints/{savename}_best.ckpt')
model.load_state_dict(checkpoint)
test_evaluator = Evaluator(model, processed_text_inputs, k=3)
check_metrics = test_evaluator.eval_recall(val_dataloader)
if best_val_recall == check_metrics['mean_recall']:
    print('Successfully load best checkpoint with acc {:.2f}'.format(
        100 * best_val_recall),
          flush=True)
else:
    print('Fail to load best checkpoint')
result = test_evaluator.submit(test_dataloader)

# save into the file
with open(f'results/{savename}_{best_val_recall}.txt', 'w') as writer:
    for label_list in result:
        a = [str(x) for x in label_list]
        save_str = ' '.join(a)
        writer.writelines(save_str + '\n')
print('Result Saved!', flush=True)
