import numpy as np
from tqdm import tqdm

# Pandas
import pandas as pd
pd.options.mode.chained_assignment = None  # Turn off the SettingWithCopyWarning
tqdm.pandas()

# torch
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, TokenClassificationPipeline


def train_test_pipeline():
    # Ask user for file paths
    train_file_path = input("Enter the path to the train dataset CSV file: ")
    test_file_path = input("Enter the path to the test dataset CSV file: ")

    # Ask user for models paths
    action_model_path = input("Enter the path to the action model(first model): ")
    validity_model_path = input("Enter the path to the validity model(second model): ")
    
    return train_file_path, test_file_path, action_model_path, validity_model_path


def text_for_pipeline():
    # Ask user for text
    text = input("Enter the text you want to enter to the model: ")

    return text


def predict_action(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    inputs.to(device)
    outputs = action_model(**inputs)
    predicted_label = torch.argmax(outputs.logits).item()
    return predicted_label


def predict_label(text, action):
    inputs = tokenizer(text, action, padding="max_length", add_special_tokens=True, truncation=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    logits = validity_model(**inputs).logits
    return F.softmax(logits, dim=1).tolist()[0]


def predict(text):
    action_number = predict_action(text)
    action = number_to_action[action_number]
    if action_number == 0:
        print(f'There is no action in the  transcript.')
        label = 0
        return action, label
    else:
        softmax = predict_label(text, action)
        label = 1 if softmax[1] > opt_thresh_model else 0
        if label == 1:
            print(f"The action is '{action}'.")
        else:
            print(f"The action '{action}' is not valid.")
        return action, label
    
# Params
actions_number = 39
number_to_action = {0: 'No action', 1: 'post up', 2: 'doubl team', 3: 'finger roll', 4: 'pump fake', 5: 'floater',  6: 'slam dunk',
 7: 'pick and roll', 8: 'coast to coast', 9: 'outlet pass', 10: 'fadeaway', 11: 'tip in',  12: 'alley oop',
 13: 'rainbow shot', 14: 'teardrop', 15: 'noth but net', 16: 'splash', 17: 'between the leg', 18: 'tomahawk',
 19: 'bank shot', 20: 'poster', 21: 'take it to the rack', 22: 'swish', 23: 'jab step',  24: 'give and go',
 25: 'flop', 26: 'basebal pass', 27: 'revers dunk',  28: 'step back', 29: 'fake', 30: 'backdoor',
 31: 'lob', 32: 'jam', 33: 'behind the back',  34: 'dime', 35: 'side step', 36: 'shake and bake', 37: 'no look pass', 38: 'euro step'}

train_file_path, test_file_path, action_model_path, validity_model_path = train_test_pipeline()

# Train Test
action_enrichment_df_train = pd.read_csv(train_file_path) # "/sise/home/ofirbenm/Wsc_ex1/action_enrichment_df_train.csv"
action_enrichment_df_test = pd.read_csv(test_file_path) # "/sise/home/ofirbenm/Wsc_ex1/action_enrichment_df_test.csv"

# Model Parmas
device = 'cpu'
bert_model = "bert-base-uncased"

# action_model_path = "/sise/home/ofirbenm/Wsc_ex1/bert_action_model.pth"
action_model = BertForSequenceClassification.from_pretrained(bert_model, num_labels=actions_number)
action_model.load_state_dict(torch.load(action_model_path, map_location=torch.device('cpu')), strict=False)
action_model.to(device)
action_model.eval()

# validity_model_path = "/sise/home/ofirbenm/Wsc_ex1/bert_validity_model.pth"
validity_model = BertForSequenceClassification.from_pretrained(bert_model, num_labels=2)
validity_model.load_state_dict(torch.load(validity_model_path, map_location=torch.device('cpu')), strict=False)
validity_model.to(device)
validity_model.eval()

tokenizer = AutoTokenizer.from_pretrained(bert_model)
opt_thresh_model = 0.392

while True:
    text = text_for_pipeline()
    predict(text)


