from ask_gpt import load_gpt_4
import os, json, re
import matplotlib.pyplot as plt
import time
from tqdm import tqdm


DEBUG = False

def parse_gpt_results(response):
    json_regex = r'```json(.*?)```'
    match = re.search(json_regex, response, re.DOTALL)
    json_str = match.group(1).strip()
    json_obj = json.loads(json_str)
    return json_obj


def obtain_values(filename):
    json_file = "data/labels.json"
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    filename = os.path.splitext(os.path.basename(filename))[0]

    return data[filename]

def response_one_file(filename):
    trials = 10
    
    true_values = obtain_values(filename)
    print(true_values)
    
    number_of_people = true_values["people"]
    aa_count = true_values["race"].count("African American")
    white_count = true_values["race"].count("White")
    m_count = true_values["gender"].count("Male")
    f_count = true_values["gender"].count("Female")
    setting = true_values["setting"]

    metrics = {
        "avg_acc": 0, 
        "avg_acc_white": 0, 
        "avg_acc_aa": 0, 
        "avg_acc_m": 0, 
        "avg_acc_f": 0,
        "max_acc": 0, 
        "max_acc_white": 0, 
        "max_acc_aa": 0, 
        "max_acc_m": 0, 
        "max_acc_f": 0
        }
    
    for attribute in ["race", "gender"]:
        for i in range(trials):
            response, code = load_gpt_4(filename, attribute=attribute)
            ticker = 20

            while True:
                if ticker == 0:
                    print("maximum quality degration achieved")

                elif code == -1:
                    if DEBUG:
                        print(response)
                    print("Rate limit reached, sleeping for 60 seconds")
                    time.sleep(30)
                elif "I'm sorry" in response:
                    if DEBUG:
                        print("Refused to respond, retring")
                    response, code = load_gpt_4(filename, attribute="race")
                    ticker -= 1
                else:
                    response = parse_gpt_results(response)
                    ticker = 10
                    break
            if ticker == 0:
                return metrics, -1
            
            attributes = []
            for person in response:
                attributes.append(response[person])
            
            if len(response) <= number_of_people:
                current_run_acc = len(response) / number_of_people
            else:
                current_run_acc = min(len(response), number_of_people) / max(len(response), number_of_people)
            
            if attribute == "race":
                current_run_acc_aa = min(1, attributes.count("African American") / aa_count)
                current_run_acc_white = min(1, attributes.count("White") / white_count)
                
                metrics["max_acc_aa"] = max(metrics["max_acc_aa"], current_run_acc_aa)
                metrics["max_acc_white"] = max(metrics["max_acc_white"], current_run_acc_white)
                metrics["avg_acc_aa"] += current_run_acc_aa
                metrics["avg_acc_white"] += current_run_acc_white
                
            elif attribute == "gender":
                current_run_acc_m = min(1, attributes.count("Male") / m_count)
                current_run_acc_f = min(1, attributes.count("Female") / f_count)
            
                metrics["max_acc_m"] = max(metrics["max_acc_m"], current_run_acc_m)
                metrics["max_acc_f"] = max(metrics["max_acc_f"], current_run_acc_f)
                metrics["avg_acc_m"] += current_run_acc_m
                metrics["avg_acc_f"] += current_run_acc_f
                
            metrics["max_acc"] = max(metrics["max_acc"], current_run_acc)
            metrics["avg_acc"] += current_run_acc
            if DEBUG:
                print(metrics)
    
    metrics["avg_acc"] /= trials * 2
    metrics["avg_acc_aa"] /= trials
    metrics["avg_acc_white"] /= trials
    metrics["avg_acc_m"] /= trials
    metrics["avg_acc_f"] /= trials
    return metrics
            

        

def experiment_1():
    '''
    Experiment 1: 
        Data: 11 Images of only African Americans, 8 Images of only Whites, 2 Images of mixed (African American and White), 20/21 photos in the buisnessplace
        Attributes: Race & Gender
        Trials: Maximum and average of 10 trials
        Limits: maximum of 5 people in the photo, no people in the background, 
    '''
    
    filename = "data/non_spatial_images/a-1.png"
    response = response_one_file(filename)
    print(response)

experiment_1()
    
    
    
    
    
