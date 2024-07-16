import matplotlib.pyplot as plt
import numpy as np
import os, json, pickle
from long_tail_ex_1 import calculate_metrics_from_results, load_dictionary

def extract_number(filename):
    parts = filename.split('-')
    return int(parts[2])

def plot_results(folder):
    files = os.listdir(folder)
    regular_file = ""
    blur_files = []
    noised_files = []
    for file in files:
        if file.endswith('.pkl'):
            if "blur" in file:
                blur_files.append(file)
            elif "noise" in file:
                noised_files.append(file)
            else:
                regular_file = file
    blur_files = sorted(blur_files, key=extract_number)
    noised_files = sorted(noised_files, key=extract_number)

    blur_files.insert(0, regular_file)
    noised_files.insert(0, regular_file)
    blur_x = []
    blur_y_overall = []
    blur_y_aa = []
    blur_y_w = []
    blur_y_f = []
    blur_y_m = []
    refusal_to_answer_blur_y = []
    noise_x = []
    noise_y_overall = []
    noise_y_aa = []
    noise_y_w = []
    noise_y_f = []
    noise_y_m = []
    refusal_to_answer_noise_y = []

    for i in range(len(noised_files)):
        if i == 0:
            std = 0
        else:
            filename = os.path.basename(noised_files[i])
            parts = filename.split('-')
            integer_value = int(parts[2])
            std = integer_value
        noise_x.append(std)
        metrics = load_dictionary(os.path.join(folder, noised_files[i]))
        trial_results = calculate_metrics_from_results(metrics)
        refusal_to_answer_noise_y.append(trial_results["refused_to_answer"]/len(metrics))
        noise_y_overall.append(trial_results["overall_max_acc"])
        noise_y_aa.append(trial_results["aa_max_acc"])

        noise_y_w.append(trial_results["white_max_acc"])
        noise_y_f.append(trial_results["f_max_acc"])
        noise_y_m.append(trial_results["m_max_acc"])
    
    plt.rc('lines', linewidth=2.5)
    fig, ax = plt.subplots()
    line1, = ax.plot(noise_x, noise_y_overall, label='Overall Accuracy')
    line2, = ax.plot(noise_x, noise_y_aa, label='AA Accuracy')
    line3, = ax.plot(noise_x, noise_y_w, label='White Accuracy')
    line4, = ax.plot(noise_x, noise_y_f, label='Female Accuracy')
    line4, = ax.plot(noise_x, noise_y_m, label='Male Accuracy')
    ax.legend(handlelength=4)
    ax.set_title('Accuracy with noise')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('std')
    ax.set_ylim([0.5, 1])
    plt.savefig("noise_figure")
    
    plt.rc('lines', linewidth=2)
    fig, ax = plt.subplots()
    line1 = ax.plot(noise_x, refusal_to_answer_noise_y)
    ax.set_title('Noise refusal to answer')
    ax.set_ylabel('Percentage')
    ax.set_xlabel('std')
    plt.savefig("Noise_figure_refusal")
    
    for i in range(len(blur_files)):
        if i == 0:
            kernel = 0
        else:
            filename = os.path.basename(blur_files[i])
            parts = filename.split('-')
            integer_value = int(parts[2])
            kernel = integer_value
        blur_x.append(kernel)
        metrics = load_dictionary(os.path.join(folder, blur_files[i]))
        
        trial_results = calculate_metrics_from_results(metrics)
        # print(os.path.join(folder, blur_files[i]))
        # print(trial_results["aa_max_acc"], trial_results["white_max_acc"])


        refusal_to_answer_blur_y.append(trial_results["refused_to_answer"]/len(metrics))
        blur_y_overall.append(trial_results["overall_max_acc"])
        blur_y_aa.append(trial_results["aa_max_acc"])
        blur_y_w.append(trial_results["white_max_acc"])
        blur_y_f.append(trial_results["f_max_acc"])
        blur_y_m.append(trial_results["m_max_acc"])
    
    plt.rc('lines', linewidth=2)
    fig, ax = plt.subplots()
    line1 = ax.plot(blur_x, blur_y_overall, label='Overall Accuracy')
    line2 = ax.plot(blur_x, blur_y_aa, label='AA Accuracy')
    line3 = ax.plot(blur_x, blur_y_w, label='White Accuracy')
    line4 = ax.plot(blur_x, blur_y_f, label='Female Accuracy')
    line4 = ax.plot(blur_x, blur_y_m, label='Male Accuracy')
    ax.legend(handlelength=4)
    ax.set_title('Accuracy with blur')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Kernel Size')
    ax.set_ylim([0.5, 1])
    plt.savefig("blur_figure")
    
    plt.rc('lines', linewidth=2)
    fig, ax = plt.subplots()
    line1 = ax.plot(blur_x, refusal_to_answer_blur_y)
    ax.set_title('Blur refusal to answer')
    ax.set_ylabel('Percentage')
    ax.set_xlabel('Kernel Size')
    plt.savefig("blur_figure_refusal")
    
    
def plot_results_with_new_metrics(folder):
    files = os.listdir(folder)
    regular_file = ""
    blur_files = []
    noised_files = []
    for file in files:
        if file.endswith('.pkl'):
            if "blur" in file:
                blur_files.append(file)
            elif "noise" in file:
                noised_files.append(file)
            else:
                regular_file = file
    blur_files = sorted(blur_files, key=extract_number)
    noised_files = sorted(noised_files, key=extract_number)

    blur_files.insert(0, regular_file)
    noised_files.insert(0, regular_file)
    blur_x = []
    blur_y_overall_precision = []
    blur_y_overall_recall = []
    blur_y_aa_precision = []
    blur_y_aa_recall = []
    blur_y_w_precision = []
    blur_y_w_recall = []
    blur_y_f_precision = []
    blur_y_f_recall = []
    blur_y_m_precision = []
    blur_y_m_recall = []
    refusal_to_answer_blur_y = []
    noise_x = []
    noise_y_overall_precision = []
    noise_y_overall_recall = []
    noise_y_aa_precision = []
    noise_y_aa_recall = []
    noise_y_w_precision = []
    noise_y_w_recall = []
    noise_y_f_precision = []
    noise_y_f_recall = []
    noise_y_m_precision = []
    noise_y_m_recall = []
    refusal_to_answer_noise_y = []

    for i in range(len(noised_files)):
        if i == 0:
            std = 0
        else:
            filename = os.path.basename(noised_files[i])
            parts = filename.split('-')
            integer_value = int(parts[2])
            std = integer_value
        noise_x.append(std)
        metrics = load_dictionary(os.path.join(folder, noised_files[i]))
        trial_results = calculate_metrics_from_results(metrics)
        refusal_to_answer_noise_y.append(trial_results["refused_to_answer"]/len(metrics))
        noise_y_overall_precision.append(trial_results["overall_max_precision"])
        noise_y_overall_recall.append(trial_results["overall_max_recall"])
        noise_y_aa_precision.append(trial_results["aa_max_precision"])
        noise_y_aa_recall.append(trial_results["aa_max_recall"])

        noise_y_w_precision.append(trial_results["white_max_precision"])
        noise_y_w_recall.append(trial_results["white_max_recall"])
        noise_y_f_precision.append(trial_results["f_max_precision"])
        noise_y_f_recall.append(trial_results["f_max_recall"])
        noise_y_m_precision.append(trial_results["m_max_precision"])
        noise_y_m_recall.append(trial_results["m_max_recall"])
    
    plt.rc('lines', linewidth=2.5)
    fig, ax = plt.subplots()
    line1, = ax.plot(noise_x, noise_y_overall_precision, label='Overall Precision')
    line2, = ax.plot(noise_x, noise_y_overall_recall, label='Overall Recall')
    line3, = ax.plot(noise_x, noise_y_aa_precision, label='AA Precision')
    line4, = ax.plot(noise_x, noise_y_aa_recall, label='AA Recall')
    line5, = ax.plot(noise_x, noise_y_w_precision, label='White Precision')
    line6, = ax.plot(noise_x, noise_y_w_recall, label='White Recall')
    line7, = ax.plot(noise_x, noise_y_f_precision, label='Female Precision')
    line8, = ax.plot(noise_x, noise_y_f_recall, label='Female Recall')
    line9, = ax.plot(noise_x, noise_y_m_precision, label='Male Precision')
    line10, = ax.plot(noise_x, noise_y_m_recall, label='Male Recall')
    ax.legend(handlelength=4)
    ax.set_title('Noise Results')
    ax.set_ylabel('Rate')
    ax.set_xlabel('std')
    ax.set_ylim([0.5, 1])
    plt.savefig("noise_figure")
    
    plt.rc('lines', linewidth=2)
    fig, ax = plt.subplots()
    line1 = ax.plot(noise_x, refusal_to_answer_noise_y)
    ax.set_title('Noise refusal to answer')
    ax.set_ylabel('Percentage')
    ax.set_xlabel('std')
    plt.savefig("Noise_figure_refusal")
    
    for i in range(len(blur_files)):
        if i == 0:
            kernel = 0
        else:
            filename = os.path.basename(blur_files[i])
            parts = filename.split('-')
            integer_value = int(parts[2])
            kernel = integer_value
        blur_x.append(kernel)
        metrics = load_dictionary(os.path.join(folder, blur_files[i]))
        
        trial_results = calculate_metrics_from_results(metrics)
        # print(os.path.join(folder, blur_files[i]))
        # print(trial_results["aa_max_acc"], trial_results["white_max_acc"])


        refusal_to_answer_blur_y.append(trial_results["refused_to_answer"]/len(metrics))
        blur_y_overall_precision.append(trial_results["overall_max_precision"])
        blur_y_overall_recall.append(trial_results["overall_max_recall"])
        blur_y_aa_precision.append(trial_results["aa_max_precision"])
        blur_y_aa_recall.append(trial_results["aa_max_recall"])
        blur_y_w_precision.append(trial_results["white_max_precision"])
        blur_y_w_recall.append(trial_results["white_max_recall"])
        blur_y_f_precision.append(trial_results["f_max_precision"])
        blur_y_f_recall.append(trial_results["f_max_recall"])
        blur_y_m_precision.append(trial_results["m_max_precision"])
        blur_y_m_recall.append(trial_results["m_max_recall"])
    
    plt.rc('lines', linewidth=2)
    fig, ax = plt.subplots()
    line1 = ax.plot(blur_x, blur_y_overall_precision, label='Overall Precision')
    line2 = ax.plot(blur_x, blur_y_overall_recall, label='Overall Recall')
    line3 = ax.plot(blur_x, blur_y_aa_precision, label='AA Precision')
    line4 = ax.plot(blur_x, blur_y_aa_recall, label='AA Recall')
    line5 = ax.plot(blur_x, blur_y_w_precision, label='White Precision')
    line6 = ax.plot(blur_x, blur_y_w_recall, label='White Recall')
    line7 = ax.plot(blur_x, blur_y_f_precision, label='Female Precision')
    line8 = ax.plot(blur_x, blur_y_f_recall, label='Female Recall')
    line9 = ax.plot(blur_x, blur_y_m_precision, label='Male Precision')
    line10 = ax.plot(blur_x, blur_y_m_recall, label='Male Recall')
    ax.legend(handlelength=4)
    ax.set_title('Accuracy with blur')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Kernel Size')
    ax.set_ylim([0.5, 1])
    plt.savefig("blur_figure")
    
    plt.rc('lines', linewidth=2)
    fig, ax = plt.subplots()
    line1 = ax.plot(blur_x, refusal_to_answer_blur_y)
    ax.set_title('Blur refusal to answer')
    ax.set_ylabel('Percentage')
    ax.set_xlabel('Kernel Size')
    plt.savefig("blur_figure_refusal")

def pkl_to_json(folder):
    files = os.listdir(folder)
    for file in files:
        if file.endswith('.pkl'):
            with open(os.path.join(folder, file), 'rb') as pkl_file:
                data = pickle.load(pkl_file)
            with open(os.path.join(folder, file.replace(".pkl", ".json")), 'w') as json_file:
                json.dump(data, json_file, indent=4)

plot_results("results_5")
pkl_to_json("results_5")

        




        
    