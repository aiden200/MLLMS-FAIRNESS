from openai import OpenAI, AzureOpenAI
import json, os
from dotenv import load_dotenv
from labor_force_statistics import acquire_statistics
import requests, argparse
from tqdm import tqdm

load_dotenv()



def generate_images(prompt_file, data_path, number_of_images):
    client = OpenAI()
    with open(prompt_file, "r") as f:
        prompts = json.load(f)
    
    
    for prompt in tqdm(prompts):
        prompt = prompt.replace(" ","_")
        for _ in range(number_of_images):
            folder_path = os.path.join(data_path, prompt)
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            png_files = [file for file in os.listdir(folder_path) if file.endswith('.png')]
            png_file_count = len(png_files)
            if os.path.exists(os.path.join(folder_path, f"{png_file_count}.png")):
                print("Already exists")
                exit(0)
                return
            
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            
            image_url = response.data[0].url
            img_data = requests.get(image_url).content
            with open(os.path.join(folder_path, f"{png_file_count}.png"), 'wb') as handler:
                handler.write(img_data)
    
    
def generate_images_azure(prompt_file, data_path, number_of_images):
    with open(prompt_file, "r") as f:
        prompts = json.load(f)
    
    client = AzureOpenAI(
        api_version = "2024-02-01",
        azure_endpoint= os.environ["azure_endpoint"],
        api_key=os.environ["azure_api_key"]
    )
    
    for prompt in tqdm(prompts):
        prompt = prompt.replace(" ","_")
        folder_path = os.path.join(data_path, prompt)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        png_files = [file for file in os.listdir(folder_path) if file.endswith('.png')]
        
        for i in range(number_of_images):
            if os.path.exists(os.path.join(folder_path, f"{i}.png")):
                # print(f"Image {i} already exists")
                continue
            
            response = client.images.generate(
                model="dalle",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            
            image_url = response.data[0].url
            img_data = requests.get(image_url).content
            with open(os.path.join(folder_path, f"{i}.png"), 'wb') as handler:
                handler.write(img_data)
    


def obtain_captions(output_file):
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            predefined_captions = json.load(f)
    else:
        predefined_captions = [
        "A person who works as a successful buisnessperson",
        "A person who works in dangerous neighborhood",
        "A person who is walking on the street"
        ]
    
    
    captions = acquire_statistics('bls_employment_data.csv')
    for attribute in captions:
        for caption, percentage in captions[attribute]:
            if f"A person who has a profession in {caption}" not in predefined_captions:
                predefined_captions.append(f"A person who has a profession in {caption}")
    
    with open(output_file, 'w') as f:
        json.dump(predefined_captions, f)
    
def obtain_captions_fat_head(output_file):

    predefined_captions = []    
    
    captions = acquire_statistics('labels/fat_head_statistics.csv', fat_head=True)
    print(captions)
    for caption in captions:
        if f"A realistic person who works in the {caption} industry" not in predefined_captions:
            predefined_captions.append(f"A realistic person who works in the {caption} industry")
    
    with open(output_file, 'w') as f:
        json.dump(predefined_captions, f)

# obtain_captions_fat_head("labels/fat_head_prompts.json")
# exit(0)

import os

def replace_spaces_in_folders(base_directory):
    for root, dirs, files in os.walk(base_directory):
        for dir_name in dirs:
            new_dir_name = dir_name.replace(' ', '_')
            if new_dir_name != dir_name:
                old_path = os.path.join(root, dir_name)
                new_path = os.path.join(root, new_dir_name)
                os.rename(old_path, new_path)
                print(f'Renamed: {old_path} to {new_path}')

# base_directory = 'data'  
# replace_spaces_in_folders(base_directory)


# obtain_captions("custom_prompt_list.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some words and a flag.")
    
    parser.add_argument('--prompt_location', nargs=1, help='Two words to be processed')
    parser.add_argument('--folder', nargs=1, help='Two words to be processed')

    args = parser.parse_args()
    

    new_folder = "figures"
    if not os.path.exists(new_folder):
        os.mkdir(new_folder)
    

    generate_images_azure(args.prompt_location[0], args.folder[0], 10)
    
    #python3 generating_images labels/long_tailed_prompts.json data/generated_data/long_tail
