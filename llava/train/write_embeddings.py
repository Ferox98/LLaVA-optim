import torch 
import json
from PIL import Image
from typing import Dict 
from transformers import AutoProcessor, CLIPVisionModel, CLIPImageProcessor
from tqdm import tqdm 

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def write_embeddings():
    device = torch.device('cuda:0')
    model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336").to(device)
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    data_dict = json.load(open("playground/data/llava_v1_5_mix665k.json", "r"))
    for i in tqdm(range(len(data_dict))):
        if 'image' in data_dict[i]:
            # get the image 
            img_url = data_dict[i]['image']
            img = Image.open(f'playground/data/{img_url}').convert('RGB')
            # get embeddings from vision encoder 
            img = expand2square(img, tuple(int(x*255) for x in processor.image_mean))
            img = processor(images=img, return_tensors="pt")['pixel_values'].to(device)
            embeddings = model(img)
            last_hidden_state = embeddings.last_hidden_state
            # write to disk
            embedding_dir = f'playground/embeddings/{"".join(img_url.split("/"))[:-4]}.pt'
            torch.save(last_hidden_state, embedding_dir)
            # update dictionary
            data_dict[i]['embeddings'] = embedding_dir
    # save to disk
    json.dump(data_dict, "playground/data/llava_v1_5_mix665k_embeddings.json")
        
        

