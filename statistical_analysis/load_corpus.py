import os

import html
import re
from bs4 import BeautifulSoup
import json
import pandas as pd



import os

import html
import re
from bs4 import BeautifulSoup
import json
import pandas as pd




def find_channel_prefixes(folder_path, suffixes = ["", "_labels", "_labels_controverse"],extension = ".json"):
    """
    Returns the channel prefixes used in the label json document triplets.
    """

    files = os.listdir(folder_path)
    file_set = set(files)
    full_suffixes = [s+extension for s in suffixes]
    base_names = {
        file[:-len(suffix)]
        for file in files
        for suffix in full_suffixes
        if file.endswith(suffix) and all((file[:-len(suffix)]+s) in file_set for s in full_suffixes)
    }

    return list(base_names)



def get_labels(folder_path, suffix):
    files = os.listdir(folder_path)
    label_files = [file for file in files if file.endswith(suffix)]
    dicts = []
    for file in label_files:
        with open(folder_path+'/'+file, 'r', encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    dicts.append(data)
                except json.JSONDecodeError:
                    print("Error reading file: ", file_path)
    fused_dict = {}
    for elt in dicts:
        for key, value in elt.items():
            actual_text = key.split(":", 2)[-1]
            fused_dict[clean_corpus(actual_text)] = value       
    return fused_dict    
  

def dict_to_df_updated(dict_controversy, dict_thematic):
    df = pd.DataFrame({
    "Comment": list(dict_controversy.keys()),
    "Controversy": list(dict_controversy.values()),
    "Thematic": [dict_thematic[k] for k in dict_controversy.keys()]
    })
    df = df[df["Controversy"].isin([0, 1])]
    df.dropna()
    return df



def clean_corpus(text):
    """
    Returns the string passed as an argument after cleaning it from html parsing mishandlings.
    """

    text = html.unescape(text)
    text = text.replace(r"\'", "'")
    soup = BeautifulSoup(text, "html.parser")
    plain_text = soup.get_text(separator=" ").strip()
    clean_text = re.sub(r'[\r\n]+', ' ', plain_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    return clean_text



def get_labels_dict(folder_path, contr, topic):

    files = os.listdir(folder_path)

    def extract(files):
        dicts = []
        for file in files:
            with open(os.path.join(folder_path, file), 'r', encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        dicts.append(data)
                    except json.JSONDecodeError:
                        print("Error reading file: ", file_path)
        return dicts


    def fuse_dict(dicts):
        fused_dict = {}
        for elt in dicts:
            for key, value in elt.items():
                actual_text = key.split(":", 2)[-1]
                fused_dict[clean_corpus(actual_text)] = value
        return fused_dict


    fused_contr_dict, fused_topic_dict = None, None

    if contr:
        contr_files = [file for file in files if file.endswith("labels_controverse.json")]
        contr_dicts = extract(contr_files)
        fused_contr_dict = fuse_dict(contr_dicts)
        

    if topic:
        topic_files = [file for file in files if file.endswith("labels.json")]
        topic_dicts = extract(topic_files)
        fused_topic_dict = fuse_dict(topic_dicts)

    return [fused_contr_dict, fused_topic_dict]


def load_df(folder_path, contr=True, topic= True):
    contr_label_dict, topic_label_dict = get_labels_dict(folder_path, contr, topic)
    shared_keys = set(contr_label_dict.keys()) & set(topic_label_dict.keys())
    if not shared_keys:
        raise ValueError("No shared texts between controversy and topic labels!")

    data = [{
        "text": text,
        "label": contr_label_dict[text],
        "topic": topic_label_dict[text]
    } for text in shared_keys]

    df = pd.DataFrame(data)

    df = df[df["label"].isin([0, 1])]
    df = df[df["topic"].notnull()]

    return df







