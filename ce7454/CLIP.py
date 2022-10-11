from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=[image, image, image], return_tensors="pt", padding=True)
print("inputs: ", inputs)
print("inputs['input_ids'].size(): ", inputs['input_ids'].size())
print("inputs['attention_mask'].size(): ", inputs['attention_mask'].size())
print("inputs['pixel_values'].size(): ", inputs['pixel_values'].size())
print("inputs: ", inputs.keys())
# train_dataloader = DataLoader(inputs['pixel_values'],
                              # batch_size=1,
                              # shuffle=True,
                              # num_workers=8)

# for id, batch in enumerate(train_dataloader):
#     print("id: ", id, "batch: ", batch)
# train_dataiter = iter(train_dataloader)
# batch = next(train_dataiter)
# print("batch: ", batch)



outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
print("logits_per_image.size(): ", logits_per_image.size())
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
print("probs.size(): ", probs.size())
