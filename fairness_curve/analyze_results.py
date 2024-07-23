import matplotlib.pyplot as plt
import numpy as np
import os, json, pickle
from long_tail_ex_1 import load_dictionary, calculate_precision_recall
import pandas as pd
import torch
import clip
from PIL import Image
from tqdm import tqdm
import argparse

from scipy.interpolate import make_interp_spline


def extract_number(filename):
    parts = filename.split('-')
    return int(parts[2])


def plot_bias_results_labor_statistics(resulting_folder, race_statistics, gender_statistics, labor_statistics, photos_folder):
    df = pd.read_csv(labor_statistics)
    other_race_columns = df.columns[4:]
    df['Other Races'] = df[other_race_columns].agg("sum", axis=1)
    
    resulting_statistics = {}
    
    clip_scores = clip_analysis(photos_folder, race_statistics, labor_statistics)

    with open(race_statistics, 'r') as f:
        race_statistics = json.load(f)
    with open(gender_statistics, 'r') as f:
        gender_statistics = json.load(f)
    
    
    for prompt in race_statistics:
        occupation = prompt.split("A person who has a profession in ")[-1]
        occupation_statistic = df[df['Occupation'] == occupation]
        if not occupation_statistic.empty:
            actual_white_percentage = occupation_statistic.White
            actual_other_percentage = 1 - occupation_statistic.White
            actual_f_percentage = occupation_statistic.Women
            actual_m_percentage = 1 - occupation_statistic.Women
            
            model_race_statistics = race_statistics[prompt]
            model_gender_statistics = gender_statistics[prompt]
            
            model_m_percentage = model_gender_statistics.count("male") / (model_gender_statistics.count("male") + model_gender_statistics.count("female"))
            model_f_percentage = model_gender_statistics.count("female") / (model_gender_statistics.count("male") + model_gender_statistics.count("female"))
            model_white_percentage = model_race_statistics.count("w") / (model_race_statistics.count("w") + model_race_statistics.count("n"))
            model_other_percentage = model_race_statistics.count("n") / (model_race_statistics.count("w") + model_race_statistics.count("n"))

            resulting_statistics[occupation] = {
                "awp" : actual_white_percentage.iloc[0]/100,
                "aop" : actual_other_percentage.iloc[0]/100,
                "afp" : actual_f_percentage.iloc[0]/100,
                "amp" : actual_m_percentage.iloc[0]/100,
                "mmp" : model_m_percentage,
                "mfp" : model_f_percentage,
                "mwp" : model_white_percentage,
                "mop" : model_other_percentage
            }
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 15), sharex=True)

    resulting_statistics = {k: v for k, v in sorted(resulting_statistics.items(), key=lambda item: item[1]["afp"])}
    clip_scores_list = []
    occupations_list = []
    
    for i, occupation in enumerate(resulting_statistics):
        axs[0].plot([resulting_statistics[occupation]["afp"], resulting_statistics[occupation]["mfp"]], [occupation, occupation], 'k-', marker='o')
        axs[0].plot(resulting_statistics[occupation]["afp"], occupation, 'o', color='blue', label='Self-identification' if i == 0 else "")
        axs[0].plot(resulting_statistics[occupation]["mfp"], occupation, 'o', color='red', label='AI model output' if i == 0 else "")
        clip_occupation = "A_person_who_has_a_profession_in_" + occupation
        clip_occupation = clip_occupation.replace(" ", "_")
        # axs[0].plot(clip_scores[clip_occupation], occupation, 'o', color='green', label='CLIP Score' if i == 0 else "")
        clip_scores_list.append(clip_scores[clip_occupation])
        occupations_list.append(i)
    
    
    X_Y_Spline = make_interp_spline(occupations_list, clip_scores_list)
    X_ = np.linspace(0, len(clip_scores_list), 500)
    Y_ = X_Y_Spline(X_)
    axs[0].plot(Y_, X_, 'g', linewidth=2, label='Clip Scores')


    axs[0].axvline(0.5, color='gray', linestyle='--')
    axs[0].set_xlim(-0.1, 1.1)
    axs[0].set_xlabel('Percentage of Females')
    axs[0].set_title('Gender Self-identification vs AI Model Output')
    axs[0].set_yticks(range(len(resulting_statistics.keys())))
    axs[0].set_yticklabels(resulting_statistics.keys())
    axs[0].legend()

    resulting_statistics = {k: v for k, v in sorted(resulting_statistics.items(), key=lambda item: item[1]["awp"])}

    clip_scores_list = []
    occupations_list = []
    # Race chart
    for i, occupation in enumerate(resulting_statistics):
        axs[1].plot([resulting_statistics[occupation]["awp"], resulting_statistics[occupation]["mwp"]], [occupation, occupation], 'k-', marker='o')
        axs[1].plot(resulting_statistics[occupation]["awp"], occupation, 'o', color='blue', label='Self-identification' if i == 0 else "")
        axs[1].plot(resulting_statistics[occupation]["mwp"], occupation, 'o', color='red', label='AI model output' if i == 0 else "")
        clip_occupation = "A_person_who_has_a_profession_in_" + occupation
        clip_occupation = clip_occupation.replace(" ", "_")
        axs[1].plot(clip_scores[clip_occupation], occupation, 'o', color='green', label='CLIP Score' if i == 0 else "")
        clip_scores_list.append(clip_scores[clip_occupation])
        occupations_list.append(i)
    
    
    X_Y_Spline = make_interp_spline(occupations_list, clip_scores_list)
    X_ = np.linspace(0, len(clip_scores_list), 500)
    Y_ = X_Y_Spline(X_)
    axs[1].plot(Y_, X_, 'g', linewidth=2, label='Clip Scores')
   
    # axs[1].plot(clip_scores_pred, [occupation for occupation in resulting_statistics], 'g', linewidth=2, label='Clip Scores')
        
    axs[1].axvline(0.5, color='gray', linestyle='--')
    axs[1].set_xlim(-0.1, 1.1)
    axs[1].set_xlabel('Percentage of White People')
    axs[1].set_title('Race Self-identification vs AI Model Output')
    axs[1].set_yticks(range(len(resulting_statistics.keys())))
    axs[1].set_yticklabels(resulting_statistics.keys())
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(resulting_folder, "gender_race_statistics"))
        

def clip_lookup(lookup_table, caption, filename=False):
    
    if filename:
        if ".png" in caption:
            caption = os.path.dirname(caption)
        caption = os.path.basename(caption)
    
    caption = caption.replace("_", " ")
    
    if "A person who has a profession in " not in caption:
        caption = "A person who has a profession in " + caption
    
    return lookup_table.get(caption, -1)

    
def clip_analysis(folder, race_json_file, statistics_file):

    with open(race_json_file, 'r') as f:
        race_statistics = json.load(f)
    
    lookup = {}
    scores = {}
    for i, caption in enumerate(race_statistics.keys()):
        lookup[caption] = i
    
    df = pd.read_csv(statistics_file)
    occupations = df["Occupation"]
    
    all_occupations = list(race_statistics.keys())
    
    for i in range(len(occupations)):
        occupation = occupations.iloc[i]
        if "A person who has a profession in " + occupation not in all_occupations:
            all_occupations.append("A person who has a profession in " + occupation)            


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    text = clip.tokenize(all_occupations).to(device)
    
    for caption_folder in tqdm(os.listdir(folder)):
        caption_folder_path = os.path.join(folder, caption_folder)
        if os.path.isdir(caption_folder_path):
            
            images_for_caption = []
            for image_file in os.listdir(caption_folder_path):
                if image_file.endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(caption_folder_path, image_file)
                    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                    images_for_caption.append(image)
                    
            images_for_caption = torch.cat(images_for_caption)
                    
            with torch.no_grad():
                logits_per_image, logits_per_text = model(images_for_caption, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            
            
            caption_index = clip_lookup(lookup, caption_folder_path, filename=True)
            
            if caption_index != -1:
                total = 0
                for i in range(len(probs)):
                    total += probs[i][caption_index]
                scores[caption_folder] = round(total/len(probs), 5)

            else:
                print(f"Caption not found for image: {image_path}")    
    
    return scores
 
def plot_results_with_new_metrics(folder, new_folder):
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
        trial_results = calculate_precision_recall(metrics)

        refusal_to_answer_noise_y.append(trial_results["refused_to_respond"]/(len(metrics)*2))
        noise_y_overall_precision.append(trial_results["precision_total"])
        noise_y_overall_recall.append(trial_results["recall_total"])
        noise_y_aa_precision.append(trial_results["precision_aa"])
        noise_y_aa_recall.append(trial_results["recall_aa"])

        noise_y_w_precision.append(trial_results["precision_white"])
        noise_y_w_recall.append(trial_results["recall_white"])
        noise_y_f_precision.append(trial_results["precision_female"])
        noise_y_f_recall.append(trial_results["recall_female"])
        noise_y_m_precision.append(trial_results["precision_male"])
        noise_y_m_recall.append(trial_results["recall_male"])
    
    plt.rc('lines', linewidth=2.5)
    fig, ax = plt.subplots()
    line1, = ax.plot(noise_x, noise_y_overall_precision, label='Overall Precision')
    line3, = ax.plot(noise_x, noise_y_aa_precision, label='AA Precision')
    line5, = ax.plot(noise_x, noise_y_w_precision, label='White Precision')
    line7, = ax.plot(noise_x, noise_y_f_precision, label='Female Precision')
    line9, = ax.plot(noise_x, noise_y_m_precision, label='Male Precision')
    ax.legend(handlelength=4)
    ax.set_title('Noise Precision')
    ax.set_ylabel('Rate')
    ax.set_xlabel('std')
    ax.set_ylim([0.5, 1])
    plt.savefig(os.path.join(new_folder, "noise_figure_precision"))
    
    
    plt.rc('lines', linewidth=2.5)
    fig, ax = plt.subplots()
    line2, = ax.plot(noise_x, noise_y_overall_recall, label='Overall Recall')
    line4, = ax.plot(noise_x, noise_y_aa_recall, label='AA Recall')
    line6, = ax.plot(noise_x, noise_y_w_recall, label='White Recall')
    line8, = ax.plot(noise_x, noise_y_f_recall, label='Female Recall')
    line10, = ax.plot(noise_x, noise_y_m_recall, label='Male Recall')
    ax.legend(handlelength=4)
    ax.set_title('Noise Recall')
    ax.set_ylabel('Rate')
    ax.set_xlabel('std')
    ax.set_ylim([0.5, 1])
    plt.savefig(os.path.join(new_folder, "noise_figure"))
    
    plt.rc('lines', linewidth=2)
    fig, ax = plt.subplots()
    line1 = ax.plot(noise_x, refusal_to_answer_noise_y)
    ax.set_title('Noise refusal to answer')
    ax.set_ylabel('Percentage')
    ax.set_xlabel('std')
    plt.savefig(os.path.join(new_folder, "Noise_figure_refusal"))
    
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
        
        trial_results = calculate_precision_recall(metrics)
        # print(os.path.join(folder, blur_files[i]))
        # print(trial_results["aa_max_acc"], trial_results["white_max_acc"])


        refusal_to_answer_blur_y.append(trial_results["refused_to_respond"]/(len(metrics)*2))
        blur_y_overall_precision.append(trial_results["precision_total"])
        blur_y_overall_recall.append(trial_results["recall_total"])
        blur_y_aa_precision.append(trial_results["precision_aa"])
        blur_y_aa_recall.append(trial_results["recall_aa"])
        blur_y_w_precision.append(trial_results["precision_white"])
        blur_y_w_recall.append(trial_results["recall_white"])
        blur_y_f_precision.append(trial_results["precision_female"])
        blur_y_f_recall.append(trial_results["recall_female"])
        blur_y_m_precision.append(trial_results["precision_male"])
        blur_y_m_recall.append(trial_results["recall_male"])

    
    plt.rc('lines', linewidth=2)
    fig, ax = plt.subplots()
    line1 = ax.plot(blur_x, blur_y_overall_precision, label='Overall Precision')
    line3 = ax.plot(blur_x, blur_y_aa_precision, label='AA Precision')
    line5 = ax.plot(blur_x, blur_y_w_precision, label='White Precision')
    line7 = ax.plot(blur_x, blur_y_f_precision, label='Female Precision')
    line9 = ax.plot(blur_x, blur_y_m_precision, label='Male Precision')
    ax.legend(handlelength=4)
    ax.set_title('Blur Precision')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Kernel Size')
    ax.set_ylim([0.5, 1.01])
    plt.savefig(os.path.join(new_folder, "blur_precision"))
    
    
    plt.rc('lines', linewidth=2)
    fig, ax = plt.subplots()
    line2 = ax.plot(blur_x, blur_y_overall_recall, label='Overall Recall')
    line4 = ax.plot(blur_x, blur_y_aa_recall, label='AA Recall')
    line6 = ax.plot(blur_x, blur_y_w_recall, label='White Recall')
    line8 = ax.plot(blur_x, blur_y_f_recall, label='Female Recall')
    line10 = ax.plot(blur_x, blur_y_m_recall, label='Male Recall')
    ax.legend(handlelength=4)
    ax.set_title('Blur Recall')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Kernel Size')
    ax.set_ylim([0.5, 1.01])
    plt.savefig(os.path.join(new_folder, "blur_recall"))
    
    
    plt.rc('lines', linewidth=2)
    fig, ax = plt.subplots()
    line1 = ax.plot(blur_x, refusal_to_answer_blur_y)
    ax.set_title('Blur refusal to answer')
    ax.set_ylabel('Percentage')
    ax.set_xlabel('Kernel Size')
    plt.savefig(os.path.join(new_folder, "blur_figure_refusal"))

def pkl_to_json(folder):
    files = os.listdir(folder)
    for file in files:
        if file.endswith('.pkl'):
            with open(os.path.join(folder, file), 'rb') as pkl_file:
                data = pickle.load(pkl_file)
            with open(os.path.join(folder, file.replace(".pkl", ".json")), 'w') as json_file:
                json.dump(data, json_file, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some words and a flag.")
    
    parser.add_argument('--folder', nargs=1, metavar=('word1'), help='Two words to be processed')
    parser.add_argument('--metrics', action='store_true', help='An optional flag with no arguments')
    parser.add_argument('--labor', action='store_true', help='An optional flag with no arguments')
    parser.add_argument('--tail', action='store_true', help='An optional flag with no arguments')

    args = parser.parse_args()
    

    new_folder = "figures"
    if not os.path.exists(new_folder):
        os.mkdir(new_folder)
    
    if args.labor:
        gender_json_file = "biases/BLIP2/results/blip2_results.json"
        race_json_file = "labels/race_results.json"
        labor_statistics = "labels/bls_employment_data.csv"
        photos_folder = "data/generated_data/fat_tail"
        if args.tail: # need to fix this in general
            gender_json_file = "biases/BLIP2/results/blip2_results.json" #fix
            race_json_file = "labels/race_results.json"
            labor_statistics = "labels/bls_employment_data.csv"
            photos_folder = "data/generated_data/fat_tail"

        
        plot_bias_results_labor_statistics(new_folder, race_json_file, gender_json_file, labor_statistics, photos_folder)
        
    if args.folder and not os.path.exists(args.folder):
        print(f"Folder: {args.folder} does not exist")
        if args.metrics:
            plot_results_with_new_metrics(args.folder, new_folder)
    

            
    # clip_analysis("None_exist", "biases/data", race_json_file)



        
    