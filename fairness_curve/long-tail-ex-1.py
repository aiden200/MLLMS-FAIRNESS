from ask_gpt import load_gpt_4
import os, json, re
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

def parse_gpt_results(response):
    json_regex = r'```json(.*?)```'
    match = re.search(json_regex, response, re.DOTALL)
    json_str = match.group(1).strip()
    json_obj = json.loads(json_str)
    return json_obj

def experiment_1():
    '''
    Experiment 1: 
        Data: 11 Images of only African Americans, 8 Images of only Whites, 2 Images of mixed (African American and White)
        Attributes: Race & Gender
        Trials: Maximum and average of 10 trials
        Limits: maximum of 5 people in the photo, no people in the background
    '''
    
    
    
    response, code = load_gpt_4(filename, attribute=attribute)
    ticker = 10
    stop_trial = False
    while True:
        if ticker == 0:
            print("maximum quality degration achieved")
            stop_trial = True
            break
        elif code == -1:
            print("Rate limit reached, sleeping for 60 seconds")
            time.sleep(60)
        elif "I'm sorry" in response:
            response, code = load_gpt_4(filename, attribute="race")
            ticker -= 1
        else:
            response = parse_gpt_results(response)
            ticker = 10
            break
    
