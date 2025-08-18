import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from statistical_analysis.analysis_df import *
from statistical_analysis.statistics import get_stats_table
from augmentation_methods.crossover_method.crossover_generate import generate_sa, generate_louvain
from utils.plot_utils import *


# hardcore paths to be adapted
STAY_DATASET_PATH = "./data/dataframes/dataset.csv"
CLEAN_STAY_DATASET_PATH = "./data/dataframes/clean_dataset.csv"
OUTPUT_PATH = "./output"

os.makedirs(OUTPUT_PATH, exist_ok=True) 
stay_df = pd.read_csv(STAY_DATASET_PATH)
clean_stay_df = pd.read_csv(CLEAN_STAY_DATASET_PATH)



def main():
    pass



    

if __name__ == "__main__":
    main()
