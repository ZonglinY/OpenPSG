import io
import json
import logging
import os

import torch
import torchvision.transforms as trn
from PIL import Image, ImageFile
from torch.utils.data import Dataset

# to fix "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


def get_transforms(stage: str):
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    if stage == 'train':
        return trn.Compose([
            Convert('RGB'),
            trn.Resize((1333, 800)),
            trn.RandomHorizontalFlip(),
            trn.RandomCrop((1333, 800), padding=4),
            trn.ToTensor(),
            trn.Normalize(mean, std),
        ])

    elif stage in ['val', 'test']:
        return trn.Compose([
            Convert('RGB'),
            trn.Resize((1333, 800)),
            trn.ToTensor(),
            trn.Normalize(mean, std),
        ])


class PSGClsDataset(Dataset):
    def __init__(
        self,
        stage,
        root='./data/coco/',
        num_classes=56,
        clip_processor=None
    ):
        super(PSGClsDataset, self).__init__()
        assert clip_processor != None
        with open('./data/psg/psg_cls_basic.json') as f:
            dataset = json.load(f)
        self.imglist = [
            d for d in dataset['data']
            if d['image_id'] in dataset[f'{stage}_image_ids']
        ]
        # print("self.imglist[0].size(): ", self.imglist[0].size())
        self.root = root
        self.transform_image = get_transforms(stage)
        self.num_classes = num_classes
        self.clip_processor = clip_processor
        # self.processed_imglist = clip_processor(images=self.imglist, return_tensors="pt", padding=True)['pixel_values']
        # print("processed_imglist.size(): ", processed_imglist.size())

    # def get_processed_imglist(self):
    #     len_imglist = len(self.imglist)


    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        # sample = {}
        sample = self.imglist[index]
        path = os.path.join(self.root, sample['file_name'])
        try:
            with open(path, 'rb') as f:
                content = f.read()
                filebytes = content
                buff = io.BytesIO(filebytes)
                image = Image.open(buff).convert('RGB')
                # sample['data'] = self.transform_image(image)
                sample['data'] = self.clip_processor(images=[image], return_tensors="pt", padding=True)['pixel_values']
                sample['data'] = sample['data'].squeeze(0)
        except Exception as e:
            logging.error('Error, cannot read [{}]'.format(path))
            raise e
        # Generate Soft Label
        soft_label = torch.Tensor(self.num_classes)
        soft_label.fill_(0)
        soft_label[sample['relations']] = 1
        sample['soft_label'] = soft_label
        del sample['relations']
        return sample
