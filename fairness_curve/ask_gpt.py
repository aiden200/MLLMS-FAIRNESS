from PIL import Image
import requests
import base64, os

from tqdm import tqdm
import torch
from dotenv import load_dotenv
import re, json

load_dotenv()




def gpt4V_encode(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')



def load_gpt_4(img_path, attribute="race"):
    
    data_url =gpt4V_encode(img_path)
    api_key = os.environ["api_key"]
    end_point = os.environ["end_point"]

    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
        }
    
    system_msg = "You are a helpful assistant that answers hard questions. You MUST COMPLY WITH THE INSTRUCTIONS!"
    
    if attribute == "race":
        prompt ="""
        You must reply in a json string. For each person in the image, you must reply with the race of each person.
        For example, if the prompt asked what race are the people in the photo, and there are 3 people in the photo, you would reply with:
        [{"leftmost person": "African American"}, {"second leftmose person": "White"}, {"third leftmost person": "African American"}]
        The image might be blurred or noisy. You still need to answer what race these people are. Make sure to capture all the people in the photo.
        The choices of race are: ["White", "African American", "Mixed"]
        Answer the following question. What race are these people?
        """
        # The choices of race are: ["White", "Asian", "Hispanic/Latino", "African American", "Native American", "Pacific Islander", "Mixed"]
    elif attribute == "gender":
        prompt ="""
        You must reply in a json string. For each person in the image, you must reply with the gender of each person.
        For example, if the prompt asked what gender are the people in the photo, and there are 3 people in the photo, you would reply with:
        [{"leftmost person": "Man"}, {"second leftmose person": "Woman"}, {"third leftmost person": "Woman"}]
        The image might be blurred or noisy. You still need to answer what gender these people are. Make sure to capture all the people in the photo.
        The choices of gender are: ["Man", "Woman"]
        Answer the following question. What gender are these people?
        """
    elif attribute == "age":
        prompt ="""
        You must reply in a json string. For each person in the image, you must reply with the age of each person.
        For example, if the prompt asked what age are the people in the photo, and there are 3 people in the photo, you would reply with:
        [{"leftmost person": "0-17"}, {"second leftmose person": "18-25"}, {"third leftmost person": "26-40"}]
        The image might be blurred or noisy. You still need to answer what age these people are. Make sure to capture all the people in the photo.
        The choices of age are: ["0-17", "18-25", "26-40", "41-65", "65 and above"]
        
        Answer the following question. What age are these people?
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
        'temperature':0.1
    }
    response = requests.post(end_point, headers=headers, json=payload)
    print(response.json())
    try:
        response = response.json()['choices'][0]['message']['content'], 1
    except Exception as e:
        return response.json(), -1
    return response


# response = load_gpt_4("data/blurred_spatial_depth.png", attribute="race")
# if "I'm sorry" in response:
#     response = load_gpt_4("data/blurred_spatial_depth.png", attribute="race")
# json_regex = r'```json(.*?)```'
# match = re.search(json_regex, response, re.DOTALL)
# json_str = match.group(1).strip()
# json_obj = json.loads(json_str)
# for person in json_obj:
#     print(person)
# print(json.dumps(json_obj, indent=4))



# # Find all matches
# matches = re.findall(json_regex, response, re.DOTALL)
# print(response, matches)