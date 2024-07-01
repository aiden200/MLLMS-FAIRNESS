
from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration


from tqdm import tqdm
import torch
import textwrap
# grab model checkpoint from huggingface hub
from huggingface_hub import hf_hub_download
import torch
# from open_flamingo import create_model_and_transforms


device = "cuda:1" if torch.cuda.is_available() else "cpu"


def obtain_flamingo():
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="anas-awadalla/mpt-7b",
        tokenizer_path="anas-awadalla/mpt-7b",
        cross_attn_every_n_layers=4
    )


    checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-9B-vitl-mpt7b", "checkpoint.pt")
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    return model, image_processor, tokenizer

def ask_flamingo(model, image_processor, tokenizer, prompt, img):
    vision_x = [image_processor(img).unsqueeze(0)]
    vision_x = torch.cat(vision_x, dim=0)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0)

    """
    Step 3: Preprocessing text
    Details: In the text we expect an <image> special token to indicate where an image is.
    We also expect an <|endofchunk|> special token to indicate the end of the text 
    portion associated with an image.
    """
    tokenizer.padding_side = "left" # For generation padding tokens should be on the left
    lang_x = tokenizer(
        [f"<image>{prompt}<|endofchunk|>"],
        return_tensors="pt",
    )

    generated_text = model.generate(
        vision_x=vision_x,
        lang_x=lang_x["input_ids"],
        attention_mask=lang_x["attention_mask"],
        max_new_tokens=80,
        num_beams=3,
    )

    return tokenizer.decode(generated_text[0])



def obtain_llava(save_path):
    try:
        llava_model = LlavaForConditionalGeneration.from_pretrained(save_path)
        llava_processor = AutoProcessor.from_pretrained(save_path)
    except:
        llava_model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
        llava_processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    
    llava_model.to(device)

    return llava_model, llava_processor

def load_image(filepath):
    image = Image.open(filepath)
    return image

def obtain_caption(model, processor, prompt, image, max_new_tokens=500):
    prompt = f"USER: <image>\n{prompt} ASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    generate_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def ask_llava(question, filepath, max_new_tokens=500):
    llava_model, llava_processor = obtain_llava("../mllms/llava-7b")
    img = load_image(filepath)
    caption = obtain_caption(llava_model, llava_processor, question, img, max_new_tokens)
    return caption


import matplotlib.pyplot as plt
import os, math


def experiment_1():
    folder_path = "Data/race_comparison_images"
    images = []
    captions = []
    model, processor = obtain_llava("../mllms/llava-7b")

    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # add more image formats if needed
            image_path = os.path.join(folder_path, filename)
            
            # Open the image
            img = Image.open(image_path)
            images.append(img)
            
            img = load_image(image_path)
            caption = obtain_caption(model, processor, "What race is this person and why?", img, max_new_tokens=100)
            caption_width = 30
            wrapped_caption = "\n".join(textwrap.wrap(caption, width=caption_width))

            captions.append(wrapped_caption)




    # Determine the grid size
    num_images = len(images)
    grid_size = math.ceil(math.sqrt(num_images))

    # Create a plot for all images and captions
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < num_images:
            ax.imshow(images[i])
            ax.set_title(captions[i])
            ax.axis('off')
        else:
            ax.axis('off')  # Hide any unused subplots

    plt.tight_layout()
    plt.savefig('results/llava_10_images.png')
    plt.show()
    
    
def experiment_2():
    folder_path = "Data/finding_images_aa"
    images = []
    captions = []
    model, processor = obtain_llava("../mllms/llava-7b")

    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # add more image formats if needed
            image_path = os.path.join(folder_path, filename)
            
            # Open the image
            img = Image.open(image_path)
            images.append(img)
            
            img = load_image(image_path)
            caption = obtain_caption(model, processor, "What race is this person and why?", img, max_new_tokens=100)
            caption_width = 30
            wrapped_caption = "\n".join(textwrap.wrap(caption, width=caption_width))

            captions.append(wrapped_caption)




    # Determine the grid size
    num_images = len(images)
    grid_size = math.ceil(math.sqrt(num_images))

    # Create a plot for all images and captions
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < num_images:
            ax.imshow(images[i])
            ax.set_title(captions[i])
            ax.axis('off')
        else:
            ax.axis('off')  # Hide any unused subplots

    plt.tight_layout()
    plt.savefig('results/llava_10_images.png')
    plt.show()
    
experiment_2()