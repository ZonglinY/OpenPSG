from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog", "not a photo"], images=[image, image, image], return_tensors="pt", padding=True)
# inputs = processor(images=[image, image, image], return_tensors="pt", padding=True)
# print("inputs: ", inputs)
print("inputs['pixel_values'].size(): ", inputs['pixel_values'].size())
print("inputs['input_ids'].size(): ", inputs['input_ids'].size())
print("inputs['attention_mask'].size(): ", inputs['attention_mask'].size())

print("inputs: ", inputs.keys())
train_data = [inputs['pixel_values'], inputs['input_ids'], inputs['attention_mask']]
train_data = TensorDataset(*train_data)
train_sampler = RandomSampler(train_data)
# train_dataloader = DataLoader(train_data,
#                               batch_size=1,
#                               shuffle=True,
#                               num_workers=8)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=1)

# for id, batch in enumerate(train_dataloader):
#     print("len(batch): ", len(batch))
#     print("id: ", id, "batch[0].size(): ", batch[0].size(), "batch[1].size(): ", batch[1].size(), "batch[2].size(): ", batch[2].size())
# train_dataiter = iter(train_dataloader)
# batch = next(train_dataiter)
# print("batch: ", batch)



outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
print("logits_per_image.size(): ", logits_per_image.size())
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
# print("probs.size(): ", probs.size())
print("probs.size(): ", probs.size())
