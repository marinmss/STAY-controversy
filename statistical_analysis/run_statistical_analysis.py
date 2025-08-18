import pandas as pd

from analysis_df import analyze_df
from stats import get_stats_table
from freq import process_ngrams
from plot_utils import *

# importing all datasets
stay_df = pd.read_csv("../data/dataframes/dataset.csv")
contr_df = stay_df[stay_df['label']==1]
non_contr_df = stay_df[stay_df['label']==0]
blog_df = pd.read_csv("../data/dataframes/blog_df.csv")
wiki_df = pd.read_csv("../data/dataframes/wiki_df.csv")

# analyzing all datasets
contr_count_df, contr_norm_count_df = analyze_df(contr_df)
non_contr_count_df, non_contr_norm_count_df = analyze_df(non_contr_df)
blog_count_df, blog_norm_count_df = analyze_df(blog_df)
wiki_count_df, wiki_norm_count_df = analyze_df(wiki_df)

dfs = {'controversial': contr_norm_count_df,
        'non-controversial': non_contr_norm_count_df,
        'blogs': blog_norm_count_df,
        'wikipedia': wiki_norm_count_df}

# setting the output path
OUTPUT_PATH = "../output"

# wordclouds
plot_wordcloud(contr_df, 'text', OUTPUT_PATH, "contr_wordcloud.png")
plot_wordcloud(contr_df, 'text', OUTPUT_PATH, "contr_wordcloud.png")
plot_wordcloud(contr_df, 'text', OUTPUT_PATH, "contr_wordcloud.png")
plot_wordcloud(contr_df, 'text', OUTPUT_PATH, "contr_wordcloud.png")

# len plots
boxplot_from_dfs(dfs, 'nb_char', OUTPUT_PATH, 'len_strip_plot')
boxplot_from_dfs({'controversial': contr_norm_count_df, 'non-controversial': non_contr_norm_count_df}, 'nb_char', OUTPUT_PATH, 'comment_len_strip_plot')

# char plots
boxplot_from_dfs(dfs, 'nb_words', OUTPUT_PATH, 'word_strip_plot')
boxplot_from_dfs({'controversial': contr_norm_count_df, 'non-controversial': non_contr_norm_count_df}, 'nb_words', OUTPUT_PATH, 'comment_word_strip_plot')

# statistics
get_stats_table(contr_count_df, OUTPUT_PATH, 'contr_stats_table.txt')
get_stats_table(non_contr_count_df, OUTPUT_PATH, 'non_contr_stats_table.txt')
get_stats_table(blog_count_df, OUTPUT_PATH, 'blog_stats_table.txt')
get_stats_table(wiki_count_df, OUTPUT_PATH, 'wiki_stats_table.txt')

# ngrams
process_ngrams(contr_df['text'], output_path=OUTPUT_PATH, filename="contr_freq_ngrams.txt")