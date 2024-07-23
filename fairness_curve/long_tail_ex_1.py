from ask_gpt import load_gpt_4
import os, json, re, pickle, argparse
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score
import numpy as np


DEBUG = False

def parse_gpt_results(response):
    try:
        json_regex = r'```json(.*?)```'
        match = re.search(json_regex, response, re.DOTALL)
        json_str = match.group(1).strip()
        json_obj = json.loads(json_str)
        return json_obj
    except Exception as e:
        print(response)
        print(e)
        exit(0)

# example_string = """
# ('```json\n[\n  {"leftmost person": "White"},\n  {"second leftmost person": "White"},\n  {"third leftmost person": "African American"},\n  {"fourth leftmost person": "African American"}\n]\n```', 1)
# """
# print(parse_gpt_results(example_string))


def obtain_values(filename, label_filename):
    json_file = label_filename
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
    refused_to_answer = 0
    for file in metrics:
        file_results = metrics[file]
        if file_results["max_acc"] != "NA":
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
        else:
            # print(file_results)
            refused_to_answer += file_results.get("refused_to_answer", 0)
            
    
    overall_count = max(overall_count, 1e-5)
    aa_count = max(aa_count, 1e-5)
    white_count = max(white_count, 1e-5)
    m_count = max(m_count, 1e-5)
    f_count = max(f_count, 1e-5)
    
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
        "f_max_acc": round(f_max_acc, 3),
        "refused_to_answer": refused_to_answer,
        "sample_response": metrics.get("sample_response", "")
    }
    
    # print(final_results)
    return final_results
        

def new_calculate_metrics_from_results(metrics):
    overall_avg_precision = 0
    overall_avg_recall = 0
    overall_max_precision = 0
    overall_max_recall = 0
    aa_avg_precision = 0
    aa_avg_recall = 0
    aa_max_precision = 0
    aa_max_recall = 0
    white_avg_precision = 0
    white_avg_recall = 0
    white_max_precision = 0
    white_max_recall = 0
    m_avg_precision = 0
    m_avg_recall = 0
    m_max_precision = 0
    m_max_recall = 0
    f_avg_precision = 0
    f_avg_recall = 0
    f_max_precision = 0
    f_max_recall = 0
    
    overall_count = 0
    aa_count = 0
    white_count = 0
    m_count = 0
    f_count = 0
    refused_to_answer = 0
    for file in metrics:
        file_results = metrics[file]
        if file_results["max_precision"] != "NA":
            overall_count += 1
            overall_avg_precision += file_results["avg_precision"]
            overall_avg_recall += file_results["avg_recall"]
            overall_max_precision += file_results["max_precision"]
            overall_max_recall += file_results["max_recall"]
        
            if file_results["max_precision_aa"] != "NA":
                aa_count += 1
                aa_avg_precision += file_results["avg_precision_aa"]
                aa_avg_recall += file_results["avg_recall_aa"]
                aa_max_precision += file_results["max_precision_aa"]
                aa_max_recall += file_results["max_recall_aa"]
            if file_results["max_precision_white"] != "NA":
                white_count += 1
                white_avg_precision += file_results["avg_precision_white"]
                white_avg_recall += file_results["avg_recall_white"]
                white_max_precision += file_results["max_precision_white"]
                white_max_recall += file_results["max_recall_white"]
            if file_results["max_precision_m"] != "NA":
                m_count += 1
                m_avg_precision += file_results["avg_precision_m"]
                m_avg_recall += file_results["avg_recall_m"]
                m_max_precision += file_results["max_precision_m"]
                m_max_recall += file_results["max_recall_m"]
            if file_results["max_precision_f"] != "NA":
                f_count += 1
                f_avg_precision += file_results["avg_precision_f"]
                f_avg_recall += file_results["avg_recall_f"]
                f_max_precision += file_results["max_precision_f"]
                f_max_recall += file_results["max_recall_f"]
        else:
            # print(file_results)
            refused_to_answer += file_results.get("refused_to_answer", 0)
            
    
    overall_count = max(overall_count, 1e-5)
    aa_count = max(aa_count, 1e-5)
    white_count = max(white_count, 1e-5)
    m_count = max(m_count, 1e-5)
    f_count = max(f_count, 1e-5)
    
    overall_avg_precision /= overall_count
    overall_avg_recall /= overall_count
    overall_max_precision /= overall_count
    overall_max_recall /= overall_count
    aa_avg_precision /= aa_count
    aa_avg_recall /= aa_count
    aa_max_precision /= aa_count
    aa_max_recall /= aa_count
    white_avg_precision /= white_count
    white_avg_recall /= white_count
    white_max_precision /= white_count
    white_max_recall /= white_count
    m_avg_precision /= m_count
    m_avg_recall /= m_count
    m_max_precision /= m_count
    m_max_recall /= m_count
    f_avg_precision /= f_count
    f_avg_recall /= f_count
    f_max_precision /= f_count
    f_max_recall /= f_count
    
    final_results = {
        "overall_avg_precision": round(overall_avg_precision, 3),
        "overall_avg_recall": round(overall_avg_recall, 3),
        "overall_max_precision": round(overall_max_precision, 3),
        "overall_max_recall": round(overall_max_recall, 3),
        "aa_avg_precision": round(aa_avg_precision, 3),
        "aa_avg_recall": round(aa_avg_recall, 3),
        "aa_max_precision": round(aa_max_precision, 3),
        "aa_max_recall": round(aa_max_recall, 3),
        "white_avg_precision": round(white_avg_precision, 3),
        "white_avg_recall": round(white_avg_recall, 3),
        "white_max_precision": round(white_max_precision, 3),
        "white_max_recall": round(white_max_recall, 3),
        "m_avg_precision": round(m_avg_precision, 3),
        "m_avg_recall": round(m_avg_recall, 3),
        "m_max_precision": round(m_max_precision, 3),
        "m_max_recall": round(m_max_recall, 3),
        "f_avg_precision": round(f_avg_precision, 3),
        "f_avg_recall": round(f_avg_recall, 3),
        "f_max_precision": round(f_max_precision, 3),
        "f_max_recall": round(f_max_recall, 3),
        "refused_to_answer": refused_to_answer,
        "sample_response": metrics.get("sample_response", "")
    }
    
    # print(final_results)
    return final_results
        

def response_one_file(filename, sample_response = False):
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
        "avg_precision": 0, 
        "avg_recall": 0, 
        "avg_precision_white": 0, 
        "avg_recall_white": 0, 
        "avg_precision_aa": 0, 
        "avg_recall_aa": 0, 
        "avg_precision_m": 0, 
        "avg_recall_m": 0, 
        "avg_precision_f": 0,
        "avg_recall_f": 0,
        "max_precision": 0, 
        "max_recall": 0, 
        "max_precision_white": 0, 
        "max_recall_white": 0, 
        "max_precision_aa": 0, 
        "max_recall_aa": 0, 
        "max_precision_m": 0, 
        "max_recall_m": 0, 
        "max_precision_f": 0,
        "max_recall_f": 0,
        "refused_to_answer": 0,
        "sample_response": []
        }
    
    keywords_present = ["I'm sorry", "Unable", "determine"]
    keywords_absent = ["White", "African American", "Mixed", "Male", "Female"]
    # keywords_absent = ["White", "African American", "Mixed", "Male", "Female", "Other"]
    number_of_trials_race = 0
    number_of_trials_gender = 0
    
    for attribute in ["race", "gender"]:
        for i in range(trials):
            ticker = 20

            while True:
                response, code = load_gpt_4(filename, attribute=attribute)
                if ticker == 0:
                    print("maximum quality degration achieved")
                    break
                elif code == -1:
                    if DEBUG:
                        print(response)
                    print("Rate limit reached, sleeping for 30 seconds")
                    time.sleep(30)
                elif any(keyword in response for keyword in keywords_present) or not any(keyword in response for keyword in keywords_absent):
                    if DEBUG:
                        print(f"Refused to respond, retrying. {ticker} times left")
                        print(response)
                    ticker -= 1
                else:
                    response = parse_gpt_results(response)
                    if DEBUG:
                        print(f"RESPONSE: {response}")
                    if attribute == "race":
                        number_of_trials_race += 1
                    elif attribute == "gender":
                        number_of_trials_gender += 1
                    ticker = 10
                    break
            if ticker == 0:
                continue
            
            attributes = []
            try:
                for person in response:
                    for key in person:
                        attributes.append(person[key])
            except Exception as e:
                print(response)
                print(e)
                exit(0)
            
            metrics["sample_response"].append(response)
            
            
            tp = min(len(attributes), number_of_people)
            fp = 0 if len(attributes) <= number_of_people else len(attributes) - number_of_people
            
            precision = tp / max((tp + fp), 1e-8)
            recall = tp / max(1e-8, number_of_people)
            
            metrics["max_precision"] = max(metrics["max_precision"], precision)
            metrics["avg_precision"] += precision
            metrics["max_recall"] = max(metrics["max_recall"], recall)
            metrics["avg_recall"] += recall

            
            if attribute == "race":
                tp = min(attributes.count("African American"), max(1e-8, aa_count))
                fp = 0 if attributes.count("African American") <= aa_count else attributes.count("African American") - aa_count
                
                precision_aa = tp / max((tp + fp), 1e-8)
                recall_aa = tp / max(1e-8, aa_count)
                
                metrics["max_precision_aa"] = max(metrics["max_precision_aa"], precision_aa)
                metrics["avg_precision_aa"] += precision_aa
                metrics["max_recall_aa"] = max(metrics["max_recall_aa"], recall_aa)
                metrics["avg_recall_aa"] += recall_aa

                tp = min(attributes.count("White"), max(1e-8, white_count))
                fp = 0 if attributes.count("White") <= white_count else attributes.count("White") - white_count
                
                precision_white = tp / max((tp + fp), 1e-8)
                recall_white = tp / max(1e-8, white_count)
                
                metrics["max_precision_white"] = max(metrics["max_precision_white"], precision_white)
                metrics["avg_precision_white"] += precision_white
                metrics["max_recall_white"] = max(metrics["max_recall_white"], recall_white)
                metrics["avg_recall_white"] += recall_white
                
                
                
            elif attribute == "gender":
                tp = min(attributes.count("Male"), max(1e-8, m_count))
                fp = 0 if attributes.count("Male") <= m_count else attributes.count("Male") - m_count
                
                precision_m = tp / max((tp + fp), 1e-8)
                recall_m = tp / max(1e-8, m_count)
                
                metrics["max_precision_m"] = max(metrics["max_precision_m"], precision_m)
                metrics["avg_precision_m"] += precision_m
                metrics["max_recall_m"] = max(metrics["max_recall_m"], recall_m)
                metrics["avg_recall_m"] += recall_m

                tp = min(attributes.count("Female"), max(1e-8, f_count))
                fp = 0 if attributes.count("Female") <= f_count else attributes.count("Female") - f_count
                
                precision_f = tp / max((tp + fp), 1e-8)
                recall_f = tp / max(1e-8, f_count)
                
                metrics["max_precision_f"] = max(metrics["max_precision_f"], precision_f)
                metrics["avg_precision_f"] += precision_f
                metrics["max_recall_f"] = max(metrics["max_recall_f"], recall_f)
                metrics["avg_recall_f"] += recall_f

                
            if DEBUG:
                print(f"Metrics of iteration: {i} for attribute: {attribute}",metrics)
    
    if number_of_trials_race == 0 or number_of_trials_gender == 0:
        metrics = {
            "avg_precision": "NA", 
            "avg_recall": "NA", 
            "avg_precision_white": "NA", 
            "avg_recall_white": "NA", 
            "avg_precision_aa": "NA", 
            "avg_recall_aa": "NA", 
            "avg_precision_m": "NA", 
            "avg_recall_m": "NA", 
            "avg_precision_f": "NA",
            "avg_recall_f": "NA",
            "max_precision": "NA", 
            "max_recall": "NA", 
            "max_precision_white": "NA", 
            "max_recall_white": "NA", 
            "max_precision_aa": "NA", 
            "max_recall_aa": "NA", 
            "max_precision_m": "NA", 
            "max_recall_m": "NA", 
            "max_precision_f": "NA",
            "max_recall_f": "NA",
            "refused_to_answer": 1
            }
        return metrics
    
    if DEBUG:
        print("# of Trials Race & Gender:",number_of_trials_race, number_of_trials_gender)
        
    metrics["avg_precision"] /= (number_of_trials_race + number_of_trials_gender)
    metrics["avg_recall"] /= (number_of_trials_race + number_of_trials_gender)
    metrics["avg_precision_aa"] /= number_of_trials_race
    metrics["avg_recall_aa"] /= number_of_trials_race
    metrics["avg_precision_white"] /= number_of_trials_race
    metrics["avg_recall_white"] /= number_of_trials_race
    metrics["avg_precision_m"] /= number_of_trials_gender
    metrics["avg_recall_m"] /= number_of_trials_gender
    metrics["avg_precision_f"] /= number_of_trials_gender
    metrics["avg_recall_f"] /= number_of_trials_gender
    
    if aa_count == 0:
        metrics["avg_precision_aa"] = "NA"
        metrics["avg_recall_aa"] = "NA"
        metrics["max_precision_aa"] = "NA"
        metrics["max_recall_aa"] = "NA"
    if white_count == 0:
        metrics["avg_precision_white"] = "NA"
        metrics["avg_recall_white"] = "NA"
        metrics["max_precision_white"] = "NA"
        metrics["max_recall_white"] = "NA"
    if m_count == 0:
        metrics["avg_precision_m"] = "NA"
        metrics["avg_recall_m"] = "NA"
        metrics["max_precision_m"] = "NA"
        metrics["max_recall_m"] = "NA"
    if f_count == 0:
        metrics["avg_precision_f"] = "NA"
        metrics["avg_recall_f"] = "NA"
        metrics["max_precision_f"] = "NA"
        metrics["max_recall_f"] = "NA"
        
    return metrics
            

def one_file_recall_and_precision(filename, label_filename):
    trials = 3
    
    true_values = obtain_values(filename, label_filename)
    
    if DEBUG:
        print(true_values)
    
    number_of_people = true_values["people"]
    aa_count = true_values["race"].count("African American")
    white_count = true_values["race"].count("White")
    m_count = true_values["gender"].count("Male")
    f_count = true_values["gender"].count("Female")
    setting = true_values["setting"]

    metrics = {
        "number_of_people": number_of_people, 
        "aa_count": aa_count, 
        "white_count": white_count, 
        "m_count": m_count, 
        "f_count": f_count, 
        "p_number_of_people": 0, 
        "p_aa_count": 0, 
        "p_white_count": 0, 
        "p_m_count": 0,
        "p_f_count": 0,
        "refused_to_respond": 0,
        "sample_response": []
        }
    
    keywords_present = ["I'm sorry", "Unable", "determine"]
    keywords_absent = ["White", "African American", "Mixed", "Male", "Female"]
    # keywords_absent = ["White", "African American", "Mixed", "Male", "Female", "Other"]
    number_of_trials_race = 0
    number_of_trials_gender = 0
    
    for attribute in ["race", "gender"]:
        for i in range(trials):
            ticker = 15

            while True:
                response, code = load_gpt_4(filename, attribute=attribute)
                if ticker == 0:
                    print("maximum quality degration achieved")
                    break
                elif code == -1:
                    if DEBUG:
                        print(response)
                    print("Rate limit reached, sleeping for 30 seconds")
                    time.sleep(30)
                elif any(keyword in response for keyword in keywords_present) or not any(keyword in response for keyword in keywords_absent):
                    if DEBUG:
                        print(f"Refused to respond, retrying. {ticker} times left")
                        print(response)
                    ticker -= 1
                else:
                    response = parse_gpt_results(response)
                    if DEBUG:
                        print(f"RESPONSE: {response}")
                    if attribute == "race":
                        number_of_trials_race += 1
                    elif attribute == "gender":
                        number_of_trials_gender += 1
                    ticker = 10
                    break
            if ticker == 0:
                continue
            
            attributes = []
            try:
                for person in response:
                    for key in person:
                        attributes.append(person[key])
            except Exception as e:
                print(response)
                print(e)
                exit(0)
            
            metrics["sample_response"].append(response)
            
            p_number_of_people = len(attributes)
            
            if abs(number_of_people - metrics['p_number_of_people']) > abs(p_number_of_people - number_of_people):
                metrics['p_number_of_people'] = p_number_of_people

            if attribute == "race":
                p_aa_count = attributes.count("African American")
                p_white_count = attributes.count("White")
                
                if abs(aa_count - metrics['p_aa_count']) > abs(p_aa_count - aa_count):
                    metrics['p_aa_count'] = p_aa_count
                if abs(white_count - metrics['p_white_count']) > abs(p_white_count - white_count):
                    metrics['p_white_count'] = p_white_count
                
                
            elif attribute == "gender":
                p_m_count = attributes.count("Male")
                p_f_count = attributes.count("Female")
                
                if abs(m_count - metrics['p_m_count']) > abs(p_m_count - m_count):
                    metrics['p_m_count'] = p_m_count
                if abs(f_count - metrics['p_f_count']) > abs(p_f_count - f_count):
                    metrics['p_f_count'] = p_f_count
                
            if DEBUG:
                print(f"Metrics of iteration: {i} for attribute: {attribute}",metrics)
    
    
    if number_of_trials_race == 0:
        metrics["refused_to_respond"] += 1
    
    if number_of_trials_gender == 0:
        metrics["refused_to_respond"] += 1
        
    if DEBUG:
        print("# of Trials Race & Gender:",number_of_trials_race, number_of_trials_gender)
    
    return metrics
    

def custom_precision_recall(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum(np.minimum(y_true, y_pred))
    fp = np.sum(np.maximum(0, y_pred - y_true))
    fn = np.sum(np.maximum(0, y_true - y_pred))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return precision, recall

def calculate_precision_recall(metrics):
    p_m_count = []
    a_m_count = []
    p_f_count = []
    a_f_count = []
    p_aa_count = []
    a_aa_count = []
    p_white_count = []
    a_white_count = []
    p_count = []
    a_count = []
    refused_to_respond = 0
    for file in metrics:
        file_results = metrics[file]
        p_m_count.append(file_results["p_m_count"])
        a_m_count.append(file_results["m_count"])
        p_f_count.append(file_results["p_f_count"])
        a_f_count.append(file_results["f_count"])
        p_aa_count.append(file_results["p_aa_count"])
        a_aa_count.append(file_results["aa_count"])
        p_white_count.append(file_results["p_white_count"])
        a_white_count.append(file_results["white_count"])
        p_count.append(file_results["p_number_of_people"])
        a_count.append(file_results["number_of_people"])
        refused_to_respond += file_results["refused_to_respond"]
    
    # print(
    #     {
    #     "p_aa_count" : p_aa_count,
    #     "a_aa_count" : a_aa_count,
    #     "p_white_count" : p_white_count,
    #     "a_white_count" : a_white_count,
    #     "p_m_count" : p_m_count,
    #     "a_m_count" : a_m_count,
    #     "p_f_count" : p_f_count,
    #     "a_f_count" : a_f_count,
    #     "p_count" : p_count,
    #     "a_count" : a_count,
    #     "refused_to_respond": refused_to_respond
    #     }
    # )

        
    precision_aa, recall_aa = custom_precision_recall(a_aa_count, p_aa_count)

    precision_white, recall_white = custom_precision_recall(a_white_count, p_white_count)
    
    precision_male, recall_male = custom_precision_recall(a_m_count, p_m_count)
    
    precision_female, recall_female = custom_precision_recall(a_f_count, p_f_count)
    
    precision_total, recall_total = custom_precision_recall(a_count, p_count)
    

    return {
        "precision_aa" : round(precision_aa, 3),
        "recall_aa" : round(recall_aa, 3),
        "precision_white" : round(precision_white, 3),
        "recall_white" : round(recall_white, 3),
        "precision_male" : round(precision_male, 3),
        "recall_male" : round(recall_male, 3),
        "precision_female" : round(precision_female, 3),
        "recall_female" : round(recall_female, 3),
        "precision_total" : round(precision_total, 3),
        "recall_total" : round(recall_total, 3),
        "refused_to_respond": refused_to_respond
    }

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

def experiment_1(folder, output_folder):
    '''
    Experiment 1: 
        Data: 11 Images of only African Americans, 8 Images of only Whites, 2 Images of mixed (African American and White), 20/21 photos in the buisnessplace
        Attributes: Race & Gender
        Trials: Maximum and average of 10 trials
        Limits: maximum of 5 people in the photo, no people in the background, 
    '''
    files = os.listdir(folder)
    test_files = [file for file in files if file.endswith('.png')]
    pickle_name = os.path.join(output_folder, folder.replace("/", "-") + ".pkl")
    print(pickle_name)
    
    done = load_dictionary(pickle_name)
    if DEBUG:
        results = new_calculate_metrics_from_results(done)
        print("Current results:", results)

    for file in tqdm(test_files):
        if file not in done:
            response = response_one_file(os.path.join(folder, file))
            done[file] = response
            save_dictionary(done, pickle_name)
            print(done)
        else:
            print(f"{file} in dic, skipping")
    
    results = new_calculate_metrics_from_results(done)
    # print(json.dumps(done, indent=4))
    print(json.dumps(results, indent=4))
    
def long_tailed_experiment(folder, output_folder, label_filename):
    
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    files = os.listdir(folder)
    test_files = [file for file in files if file.endswith('.png')]
    pickle_name = os.path.join(output_folder, folder.replace("/", "-") + ".pkl")
    print(pickle_name)
    
    done = load_dictionary(pickle_name)
    if DEBUG:
        results = calculate_precision_recall(done)
        print("Current results:", results)

    for file in tqdm(test_files):
        if file not in done:
            response = one_file_recall_and_precision(os.path.join(folder, file), label_filename)
            done[file] = response
            save_dictionary(done, pickle_name)
            print(done)
        else:
            print(f"{file} in dic, skipping")
    
    results = calculate_precision_recall(done)
    # print(json.dumps(done, indent=4))
    print(json.dumps(results, indent=4))

            
    
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some words and a flag.")
    
    parser.add_argument('--words', nargs=2, metavar=('word1', 'word2'), help='Two words to be processed')
    parser.add_argument('--flag', action='store_true', help='An optional flag with no arguments')
    parser.add_argument('--all', action='store_true', help='An optional flag with no arguments')
    parser.add_argument('--w', action='store_true', help='An optional flag with no arguments')
    parser.add_argument('--p', action='store_true', help='An optional flag with no arguments')

    args = parser.parse_args()
    
    output_folder = "results/precision_recall"
    
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    
    if args.all:
        label_filename = "labels/labels.json"
        long_tailed_experiment("data/non_spatial_images", output_folder, label_filename)
        for edit in ["blur", "noise"]:
            blur_range = ["5", "11", "17", "23", "29", "35", "41"]
            noise_range = ["15", "25", "40", "60", "70", "80"]
            if edit == "blur":
                for number in blur_range:
                    print(f"Running {edit} {number}")
                    long_tailed_experiment(f"data/{edit}-{number}-non_spatial_images", output_folder, label_filename)
            else:
                for number in noise_range:
                    print(f"Running {edit} {number}")
                    long_tailed_experiment(f"data/{edit}-{number}-non_spatial_images", output_folder, label_filename)
    elif args.w:
        label_filename = "labels/water_labels.json"
        output_folder = "results/water_precision_recall"
        long_tailed_experiment("data/drinking_water/no_noise", output_folder, label_filename)
        for edit in ["blur", "noise"]:
            blur_range = ["5", "11", "17", "23", "29", "35", "41"]
            noise_range = ["15", "25", "40", "60", "70", "80"]
            if edit == "blur":
                for number in blur_range:
                    print(f"Running {edit} {number}")
                    long_tailed_experiment(f"data/drinking_water/{edit}-{number}-no_noise", output_folder, label_filename)
            else:
                for number in noise_range:
                    print(f"Running {edit} {number}")
                    long_tailed_experiment(f"data/drinking_water/{edit}-{number}-no_noise", output_folder, label_filename)
    elif args.p:
        label_filename = "labels/poverty_labels.json"
        output_folder = "results/poverty_precision_recall"
        long_tailed_experiment("data/poverty/no_noise", output_folder, label_filename)
        for edit in ["blur", "noise"]:
            blur_range = ["5", "11", "17", "23", "29", "35", "41"]
            noise_range = ["15", "25", "40", "60", "70", "80"]
            if edit == "blur":
                for number in blur_range:
                    print(f"Running {edit} {number}")
                    long_tailed_experiment(f"data/poverty/{edit}-{number}-no_noise", output_folder, label_filename)
            else:
                for number in noise_range:
                    print(f"Running {edit} {number}")
                    long_tailed_experiment(f"data/poverty/{edit}-{number}-no_noise", output_folder, label_filename )
    elif args.flag:
        print('Flag is set')
        label_filename = "labels/labels.json"
        long_tailed_experiment("data/non_spatial_images", output_folder, label_filename)
    elif args.words:
        word1, word2 = args.words
        print(f'Word 1: {word1}')
        print(f'Word 2: {word2}')
        label_filename = "labels/labels.json"
        long_tailed_experiment(f"data/{word1}-{word2}-non_spatial_images", output_folder, label_filename)
    else:
        print("Running tests, choose flags to avoid this option")
        y_true_african_american = [2, 1, 0]
        y_pred_african_american = [2, 1, 1]
        precision, recall = custom_precision_recall(y_true_african_american, y_pred_african_american)
        print(precision, recall)
    
    
    # python3 long_tail_ex_1.py --words blur 5
    # python3 long_tail_ex_1.py --flag
    
        
    
    
    
    
    
