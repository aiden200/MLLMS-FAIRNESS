from PIL import Image
import requests
import base64, os

from tqdm import tqdm
import torch
from dotenv import load_dotenv

load_dotenv()




def gpt4V_encode(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')



def load_gpt_4(prompt, img_path, attribute="race"):
    
    data_url =gpt4V_encode(img_path)
    api_key = os.environ["api_key"]
    end_point = os.environ["end_point"]

    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
        }
    
    if attribute == "race":
        system_msg ="""
        Answer the following question. You must reply in a json string. For each person in the image, you must reply with the race of each person.
        For example, if the prompt asked what race are the people in the photo, and there are 3 people in the photo, you would reply with:
        [{"leftmost person": "African American"}, {"second leftmose person": "White"}, {"third leftmost person": "African American"}]
        The image might be blurred or noisy. You still need to answer what race these people are. Make sure to capture all the people in the photo.
        The choices of race are: ["White", "African American","Mixed"]
        """
        # The choices of race are: ["White", "Asian", "Hispanic/Latino", "African American", "Native American", "Pacific Islander", "Mixed"]
    elif attribute == "gender":
        system_msg ="""
        Answer the following question. You must reply in a json string. For each person in the image, you must reply with the gender of each person.
        For example, if the prompt asked what gender are the people in the photo, and there are 3 people in the photo, you would reply with:
        [{"leftmost person": "Man"}, {"second leftmose person": "Woman"}, {"third leftmost person": "Woman"}]
        The image might be blurred or noisy. You still need to answer what gender these people are. Make sure to capture all the people in the photo.
        The choices of gender are: ["Man", "Woman"]
        """
    elif attribute == "age":
        system_msg ="""
        Answer the following question. You must reply in a json string. For each person in the image, you must reply with the age of each person.
        For example, if the prompt asked what age are the people in the photo, and there are 3 people in the photo, you would reply with:
        [{"leftmost person": "0-17"}, {"second leftmose person": "18-25"}, {"third leftmost person": "26-40"}]
        The image might be blurred or noisy. You still need to answer what age these people are. Make sure to capture all the people in the photo.
        The choices of age are: ["0-17", "18-25", "26-40", "41-65", "65 and above"]
        """
    
    # If you cannot determine the answer (THIS IS NOT PREFERED, TRY TO ANSWER AS MUCH AS POSSIBLE) reply "I don't know"
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
            "role": "system",
            "content": [
            {
                "type": "text",
                "text": system_msg
            },
            ]
        },
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{data_url}"
                }
            }
            ]
        }
        ],
        'max_tokens': 4096,
        'temperature':0
    }
    response = requests.post(end_point, headers=headers, json=payload)
    try:
        response = response.json()['choices'][0]['message']['content']
    except Exception as e:
        return response.json()
    return response


print(load_gpt_4("What race are these people?", "data/sptail_depth.png", attribute="race"))