from ask_gpt import load_gpt_4
import os, json, re, pickle
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

# example_string = """
# ('```json\n[\n  {"leftmost person": "White"},\n  {"second leftmost person": "White"},\n  {"third leftmost person": "African American"},\n  {"fourth leftmost person": "African American"}\n]\n```', 1)
# """
# print(parse_gpt_results(example_string))


def obtain_values(filename):
    json_file = "data/labels.json"
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    filename = os.path.splitext(os.path.basename(filename))[0]

    return data[filename]


def calculate_metrics_from_results(metrics):
    overall_avg_acc = 0
    overall_max_acc = 0
    aa_avg_acc = 0
    aa_max_acc = 0
    white_avg_acc = 0
    white_max_acc = 0
    m_avg_acc = 0
    m_max_acc = 0
    f_avg_acc = 0
    f_max_acc = 0
    
    overall_count = 0
    aa_count = 0
    white_count = 0
    m_count = 0
    f_count = 0
    for file in metrics:
        file_results = metrics[file]
        overall_count += 1
        overall_avg_acc += file_results["avg_acc"]
        overall_max_acc += file_results["max_acc"]
        
        if file_results["max_acc_aa"] != "NA":
            aa_count += 1
            aa_avg_acc += file_results["avg_acc_aa"]
            aa_max_acc += file_results["max_acc_aa"]
        if file_results["max_acc_white"] != "NA":
            white_count += 1
            white_avg_acc += file_results["avg_acc_white"]
            white_max_acc += file_results["max_acc_white"]
        if file_results["max_acc_m"] != "NA":
            m_count += 1
            m_avg_acc += file_results["avg_acc_m"]
            m_max_acc += file_results["max_acc_m"]
        if file_results["max_acc_f"] != "NA":
            f_count += 1
            f_avg_acc += file_results["avg_acc_f"]
            f_max_acc += file_results["max_acc_f"]
    
    overall_avg_acc /= overall_count
    overall_max_acc /= overall_count
    aa_avg_acc /= aa_count
    aa_max_acc /= aa_count
    white_avg_acc /= white_count
    white_max_acc /= white_count
    m_avg_acc /= m_count
    m_max_acc /= m_count
    f_avg_acc /= f_count
    f_max_acc /= f_count
    
    final_results = {
        "overall_avg_acc": round(overall_avg_acc, 3),
        "overall_max_acc": round(overall_max_acc, 3),
        "aa_avg_acc": round(aa_avg_acc, 3),
        "aa_max_acc": round(aa_max_acc, 3),
        "white_avg_acc": round(white_avg_acc, 3),
        "white_max_acc": round(white_max_acc, 3),
        "m_avg_acc": round(m_avg_acc, 3),
        "m_max_acc": round(m_max_acc, 3),
        "f_avg_acc": round(f_avg_acc, 3),
        "f_max_acc": round(f_max_acc, 3)
    }
    

    return final_results
        


def response_one_file(filename):
    trials = 5
    
    true_values = obtain_values(filename)
    
    if DEBUG:
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
            ticker = 20

            while True:
                response, code = load_gpt_4(filename, attribute=attribute)
                if ticker == 0:
                    print("maximum quality degration achieved")

                elif code == -1:
                    if DEBUG:
                        print(response)
                    print("Rate limit reached, sleeping for 30 seconds")
                    time.sleep(30)
                elif "I'm sorry" in response:
                    if DEBUG:
                        print("Refused to respond, retrying")
                        print(response)
                    ticker -= 1
                else:
                    response = parse_gpt_results(response)
                    if DEBUG:
                        print(f"RESPONSE: {response}")
                    ticker = 10
                    break
            if ticker == 0:
                return metrics, -1
            
            attributes = []

            for person in response:
                for key in person:
                    attributes.append(person[key])
            
            if len(attributes) <= number_of_people:
                current_run_acc = len(attributes) / number_of_people
            else:
                current_run_acc = min(len(attributes), number_of_people) / max(len(attributes), number_of_people)
            
            if attribute == "race":
                current_run_acc_aa = min(1, attributes.count("African American") / max(1e-8, aa_count))
                current_run_acc_white = min(1, attributes.count("White") / max(1e-8, white_count))
                
                metrics["max_acc_aa"] = max(metrics["max_acc_aa"], current_run_acc_aa)
                metrics["max_acc_white"] = max(metrics["max_acc_white"], current_run_acc_white)
                metrics["avg_acc_aa"] += current_run_acc_aa
                metrics["avg_acc_white"] += current_run_acc_white
                
            elif attribute == "gender":
                current_run_acc_m = min(1, attributes.count("Male") / max(1e-8, m_count))
                current_run_acc_f = min(1, attributes.count("Female") / max(1e-8, f_count))
            
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
    
    if aa_count == 0:
        metrics["avg_acc_aa"] = "NA"
        metrics["max_acc_aa"] = "NA"
    if white_count == 0:
        metrics["avg_acc_white"] = "NA"
        metrics["max_acc_white"] = "NA"
    if m_count == 0:
        metrics["avg_acc_m"] = "NA"
        metrics["max_acc_m"] = "NA"
    if f_count == 0:
        metrics["avg_acc_f"] = "NA"
        metrics["max_acc_f"] = "NA"
        
    return metrics
            

def save_dictionary(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def load_dictionary(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        return data
    else:
        return {}

def experiment_1(folder):
    '''
    Experiment 1: 
        Data: 11 Images of only African Americans, 8 Images of only Whites, 2 Images of mixed (African American and White), 20/21 photos in the buisnessplace
        Attributes: Race & Gender
        Trials: Maximum and average of 10 trials
        Limits: maximum of 5 people in the photo, no people in the background, 
    '''
    files = os.listdir(folder)
    test_files = [file for file in files if file.endswith('.png')]
    pickle_name = folder.replace("/", "-") + ".pkl"
    
    done = load_dictionary(pickle_name)

    for file in tqdm(test_files):
        if file not in done:
            response = response_one_file(os.path.join(folder, file))
            done[file] = response
            save_dictionary(done, pickle_name)
            print(done)
        else:
            print(f"{file} in dic, skipping")
    
    results = calculate_metrics_from_results(done)
    print(json.dumps(done, indent=4))
    print(json.dumps(results, indent=4))

            
    
            
        

experiment_1("data/non_spatial_images/")
    
    
    
    
    
