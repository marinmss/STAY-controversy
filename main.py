import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from statistical_analysis.analysis_df import *
from statistical_analysis.stats import get_stats_table
from statistical_analysis.plot_utils import *

from augmentation_methods.crossover_method.crossover_generate import generate_sa, generate_louvain
from augmentation_methods.prompting_method.prompt_generation import generate_zero_shot

from finetuning_pipeline.cross_validation.configurations import *
from finetuning_pipeline.cross_validation.finetuner import finetune_cv
from finetuning_pipeline.cross_validation.tester import run_test




# STAY datapaths
STAY_DATASET_PATH = "./data/dataframes/dataset.csv"
CLEAN_STAY_DATASET_PATH = "./data/dataframes/clean_dataset.csv"

# STAY data extraction
STAY_DF = pd.read_csv(STAY_DATASET_PATH)
CLEAN_STAY_DF = pd.read_csv(CLEAN_STAY_DATASET_PATH)

# output path definition
OUTPUT_PATH = "./output"
os.makedirs(OUTPUT_PATH, exist_ok=True) 



def main():

    # STATISTICAL ANALYSIS ==============================================
    # corpora definition
    contr_df = STAY_DF[STAY_DF['label']==1]
    non_contr_df = STAY_DF[STAY_DF['label']==0]

    blog_df = pd.read_csv("./data/dataframes/blog_df.csv")
    wiki_df = pd.read_csv("./data/dataframes/wiki_df.csv")

    # count analysis and normalization
    contr_count_df, contr_norm_count_df = analyze_df(contr_df)
    non_contr_count_df, non_contr_norm_count_df = analyze_df(non_contr_df)
    blog_count_df, blog_norm_count_df = analyze_df(blog_df)
    wiki_count_df, wiki_norm_count_df = analyze_df(wiki_df)

    normalized_corpora = {'controversial': contr_norm_count_df,
                'non-controversial': non_contr_norm_count_df,
                'blogs': blog_norm_count_df,
                'wikipedia': wiki_norm_count_df}

    # wordclouds
    plot_wordcloud(contr_df, 'text', OUTPUT_PATH, "contr_wordcloud.png")
    plot_wordcloud(contr_df, 'text', OUTPUT_PATH, "contr_wordcloud.png")
    plot_wordcloud(contr_df, 'text', OUTPUT_PATH, "contr_wordcloud.png")
    plot_wordcloud(contr_df, 'text', OUTPUT_PATH, "contr_wordcloud.png")

    # len plots
    boxplot_from_dfs(normalized_corpora, 'nb_char', OUTPUT_PATH, 'len_strip_plot')
    boxplot_from_dfs({'controversial': contr_norm_count_df, 'non-controversial': non_contr_norm_count_df}, 'nb_char', OUTPUT_PATH, 'comment_len_strip_plot')

    # char plots
    boxplot_from_dfs(normalized_corpora, 'nb_words', OUTPUT_PATH, 'word_strip_plot')
    boxplot_from_dfs({'controversial': contr_norm_count_df, 'non-controversial': non_contr_norm_count_df}, 'nb_words', OUTPUT_PATH, 'comment_word_strip_plot')

    # statistics
    get_stats_table(contr_count_df, OUTPUT_PATH, 'contr_stats_table.txt')
    get_stats_table(non_contr_count_df, OUTPUT_PATH, 'non_contr_stats_table.txt')
    get_stats_table(blog_count_df, OUTPUT_PATH, 'blog_stats_table.txt')
    get_stats_table(wiki_count_df, OUTPUT_PATH, 'wiki_stats_table.txt') 

    
    # BASELINE DEFINITION ===============================================
    # data splitting
    # train_df, test_df = train_test_split(CLEAN_STAY_DF, test_size=0.2, random_state=42)
    # train_df.to_csv("./data/dataframes/clean_train_df.csv")
    # test_df.to_csv("./data/dataframes/clean_test_df.csv")

    # configuration definitions
    data_config = DataConfig(train_version = "baseline",
                            test_version = "original-clean-whole",
                            train_path = "./data/dataframes/clean_train_df.csv",
                            test_path = "./data/dataframes/clean_dataset.csv")  
    training_config = TrainingConfig()
    cv_config = CVConfig()

    # logging output definition
    finetuning_output_path = "./finetuning_pipeline/finetuning_logs"
    os.makedirs(finetuning_output_path, exist_ok=True) 

    # finetuning and baseline model saving
    _, baseline_model, baseline_tokenizer,_ = finetune_cv(data_config,training_config, cv_config, finetuning_output_path)

    # ARGUMENTATIVE CROSS-OVER AUGMENTATION =============================
    # generation
    train_df = pd.read_csv("./data/dataframes/clean_train_df.csv")
    louvain_df = generate_louvain(train_df)
    # louvain_df.to_csv("./data/dataframes/louvain_df.csv")

    # baseline predictions
    louvain_predict_df = run_test(test_df=louvain_df,
                            model = baseline_model,
                            tokenizer=baseline_tokenizer,
                            batch_size=training_config.per_device_eval_batch_size,
                            output_path=OUTPUT_PATH)
    #louvain_df.to_csv("./data/dataframes/louvain_predict_df.csv")

    # finetuning
    data_config = DataConfig(train_version = "louvain",
                            test_version = "test-split-original-clean",
                            train_path = "./data/dataframes/louvain_df.csv",
                            test_path = "./data/dataframes/clean_test_df.csv")  
    training_config = TrainingConfig()
    cv_config = CVConfig()
    finetune_cv(data_config,training_config, cv_config, finetuning_output_path)


    # MISTRAL PROMPTING AUGMENTATION ====================================
    # generation
    mistral_df = generate_zero_shot(nb_batches = 240)
    #mistral_df.to_csv(os.path.join(".data/dataframes/mistral_df.csv"), index=False)

    # baseline predictions
    mistral_predict_df = run_test(test_df=mistral_df,
                            model = baseline_model,
                            tokenizer=baseline_tokenizer,
                            batch_size=training_config.per_device_eval_batch_size,
                            output_path=OUTPUT_PATH)
    #mistral_df.to_csv("./data/dataframes/mistral_predict_df.csv")

    


    

if __name__ == "__main__":
    main()
