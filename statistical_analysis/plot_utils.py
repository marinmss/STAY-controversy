import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from math import pi
import spacy

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from wordcloud import WordCloud

nlp = spacy.load("fr_core_news_sm")



def plot_heatmap(df:pd.DataFrame, column1:str, column2:str, output_path:str='./output', filename:str = 'heatmap.png'):
    expected_columns = {column1, column2}
    missing_columns = expected_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"plot_heatmap: missing required column(s): {missing_columns}")

    os.makedirs(output_path, exist_ok=True)

    ct = pd.crosstab(df[column1], df[column2])
    plt.figure(figsize=(10, 6))
    sns.heatmap(ct, annot=True, fmt='d', cmap='Greens')
    plt.title(f"Heatmap: {column1} x {column2}")
    plt.xlabel(column2)
    plt.ylabel(column1)
    plt.tight_layout()

    savepath = os.path.join(output_path, filename)
    plt.savefig(savepath, dpi=300)
    plt.close()


def plot_histogram(df:pd.DataFrame, column:str, output_path:str = './output', filename: str = "count_histogram.png"):
    expected_columns = {column}
    missing_columns = expected_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"plot_histogram: missing required column(s): {missing_columns}")

    os.makedirs(output_path, exist_ok=True)

    counts = df[column].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    bars = plt.bar(counts.index.astype(str), counts.values)

    for bar, count in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 str(count), ha='center', va='bottom', fontsize=10)

    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.ylim(0, max(counts.values) * 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    savepath = os.path.join(output_path, filename)
    plt.savefig(savepath, dpi=300)
    plt.close()


def plot_pie_chart(df:pd.DataFrame, column:str, output_path:str = './output', filename:str = "pie_chart.png"):
    expected_columns = {column}
    missing_columns = expected_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"plot_pie_chart: missing required column(s): {missing_columns}")

    os.makedirs(output_path, exist_ok=True)

    counts = df[column].value_counts().sort_index()
    labels = [f"{idx} ({count})" for idx, count in counts.items()]

    colors = ['#a8d5a3','#f4a6a6']

    plt.figure(figsize=(7, 7))
    plt.pie(counts.values, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.tight_layout()

    savepath = os.path.join(output_path, filename)
    plt.savefig(savepath, dpi=300)
    plt.close()


def label_topic_barplot(df:pd.DataFrame, output_path:str='./output', filename_label="label_barplot.png", filename_topic="topic_barplot.png"):
    expected_columns = {'label','topic'}
    missing_columns = expected_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"label_topic_barplot: missing required column(s): {missing_columns}")

    os.makedirs(output_path, exist_ok=True)

    sns.set(style="whitegrid")
    palette_keys = {'0': "#A8E6A3", '1': "#F6A6A6"}  # string keys
    topic_palette = sns.color_palette("pastel", n_colors=df['topic'].nunique())

    def annotate_with_percentage(ax, total):
        for p in ax.patches:
            height = p.get_height()
            percentage = 100 * height / total
            ax.annotate(f"{int(height)} ({percentage:.1f}%)",
                        (p.get_x() + p.get_width() / 2, height + total * 0.01),
                        ha='center', va='bottom', fontsize=9)

    # Plot 1: Controversy label
    plt.figure(figsize=(6, 4))
    ax1 = sns.countplot(x="label", data=df, palette=palette_keys)
    total_labels = len(df)
    annotate_with_percentage(ax1, total_labels)
    ax1.set_xlabel("Controversy Label (0=Non-controversial, 1=Controversial)")
    ax1.set_ylabel("Count")
    ax1.set_ylim(0, ax1.get_ylim()[1] * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, filename_label))
    plt.close()

    # Plot 2: Topic label
    plt.figure(figsize=(10, 5))
    ax2 = sns.countplot(x="topic", data=df, palette=topic_palette)
    total_topics = len(df)
    annotate_with_percentage(ax2, total_topics)
    ax2.set_xlabel("Topic Label")
    ax2.set_ylabel("Count")
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylim(0, ax2.get_ylim()[1] * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, filename_topic))
    plt.close()



def plot_wordcloud(df:pd.DataFrame, text_column:str='text', output_path:str='./output', filename:str='wordcloud.png'):
    expected_columns = {text_column}
    missing_columns = expected_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"plot_wordcloud: missing required column(s): {missing_columns}")

    os.makedirs(output_path, exist_ok=True)

    full_text = " ".join(df[text_column].dropna().astype(str))

    doc = nlp(full_text)

    filtered_tokens = [
        token.text.lower()
        for token in doc
        if not token.is_stop and token.is_alpha
    ]

    cleaned_text = " ".join(filtered_tokens)

    wordcloud = WordCloud(
        width=1200,
        height=800,
        background_color="white",
        collocations=True,
        colormap="viridis"
    ).generate(cleaned_text)

    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")

    savepath = os.path.join(output_path, filename)
    plt.savefig(savepath, bbox_inches="tight", pad_inches=0)
    plt.close()




def plot_feature_barplot(dfs, features = None, labels = None, filename="output/graphs/bar_plot.png"):
    if labels is None or len(dfs) != len(labels):
        labels = [f"label{i+1}" for i in range(len(dfs))]
    
    if features is None:
        features = set(dfs[0].columns)
        for df in dfs[1:]:
            common_features.intersection_update(df.columns)
        features = sorted(list(common_features))
    
    normalized_data = []
    
    for df in dfs:
        normalized_values = normalize_df(df)
        normalized_data.append(normalized_values)
    

    normalized_df = pd.DataFrame(normalized_data, columns=features, index=labels)

    normalized_df_reset = normalized_df.reset_index().melt(id_vars='index', var_name='Feature', value_name='Value')
    normalized_df_reset.rename(columns={'index': 'DataFrame'}, inplace=True)
 
    plt.figure(figsize=(10, 6))
    sns.barplot(data=normalized_df_reset, x='Feature', y='Value', hue='DataFrame')
    
    plt.xlabel('Features')
    plt.ylabel('Normalized values')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()




def plot_radar_chart(dfs, features=None, labels=None, filename="output/graphs/radar_plot.png"):
    if labels is None or len(dfs) != len(labels):
        labels = [f"label{i+1}" for i in range(len(dfs))]

    if features is None:
        common_features = set(dfs[0].columns)
        for df in dfs[1:]:
            common_features.intersection_update(df.columns)
        features = sorted(list(common_features))

    normalized_dfs = [normalize_df(df)[features] for df in dfs]
    categories = features
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))  # Augmente la largeur

    for i, normalized in enumerate(normalized_dfs):
        values = normalized.values.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=labels[i], color=colors[i % len(colors)])

    for i, normalized in enumerate(normalized_dfs):
        values = normalized.values.tolist()
        values += values[:1]
        ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.legend(loc='center left', bbox_to_anchor=(1.2, 0.5))

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()




def plot_boxplot(df, column, output_path:str='./output', filename:str='boxplot.png'):
    expected_columns = {column}
    missing_columns = expected_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"plot_boxplot: missing required column(s): {missing_columns}")

    os.makedirs(output_path, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.boxplot(df[column].dropna())
    plt.xticks([1], [column])
    plt.ylabel(column)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    savepath = os.path.join(output_path, filename)
    plt.savefig(savepath, dpi=300)
    plt.close()




def stripplot_from_dfs(dfs:pd.DataFrame, column:str, output_path:str='./output', filename:str="stripplot.png"):
    os.makedirs(output_path, exist_ok=True)

    combined = []
    for i, df in enumerate(dfs):
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame {i}")
        temp = df[[column]].copy()
        temp["Source"] = f"DF{i+1}" 
        combined.append(temp)
    
    combined_df = pd.concat(combined, ignore_index=True)

    plt.figure(figsize=(8, 6))
    sns.stripplot(x="Source", y=column, data=combined_df, jitter=True, alpha=0.7)
    plt.ylabel(column)
    plt.xlabel("DataFrame")
    
    savepath = os.path.join(output_path, filename)
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close()



def boxplot_from_dfs(dfs_dict, column:str, output_path:str='./output', filename="boxplot.png"):
    os.makedirs(output_path, exist_ok=True)

    combined = []
    for label, df in dfs_dict.items():
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame labeled '{label}'")
        temp = df[[column]].copy()
        temp["Source"] = label
        combined.append(temp)
    
    combined_df = pd.concat(combined, ignore_index=True)

    plt.figure(figsize=(8, 6))
    sns.boxplot(x="Source", y=column, data=combined_df, palette="Set2")
    plt.ylabel(column)
    plt.xlabel("Data Source")
    plt.xticks(rotation=30, ha="right")

    savepath = os.path.join(output_path, filename)
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close()