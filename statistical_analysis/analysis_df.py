import pandas as pd
import spacy
import re
import emoji
import nltk

from textblob import TextBlob
from nltk.corpus import stopwords

nltk.download('stopwords')
nlp = spacy.load("fr_core_news_sm")


# -----------------------------------------------------------------------------
# TEXT FEATURE EXTRACTION FUNCTIONS
# -----------------------------------------------------------------------------

def length_df(df, column):
    """Add a column with the number of characters in the text."""
    df["nb_char"] = df[column].apply(len)
    return df


def nb_words_df(df, column):
    """Add a column with the number of words in the text."""
    df["nb_words"] = df[column].apply(lambda text: len(text.split()))
    return df


def count_pos_df(df, column, pos_tags):
    """Count occurrences of each POS tag in the text."""
    def count_pos(text, pos_tag):
        doc = nlp(text)
        return sum(1 for token in doc if token.pos_ == pos_tag)

    for pos_tag in pos_tags:
        df[pos_tag] = df[column].apply(lambda text: count_pos(text, pos_tag))
    return df


def count_punct_df(df, column, punct_list):
    """Count occurrences of each punctuation symbol in the text."""
    def count_punct(text, punct):
        if not isinstance(text, str):
            return 0
        return text.count(punct)

    for punct in punct_list:
        df[punct] = df[column].apply(lambda text: count_punct(text, punct))
    return df


def count_emojis_df(df, column):
    """Count the number of emojis in the text."""
    def count_emoji(text):
        if not isinstance(text, str):
            return 0
        return sum(1 for char in text if char in emoji.EMOJI_DATA)

    df["nb_emojis"] = df[column].apply(count_emoji)
    return df


def count_urls_df(df, column):
    """Count the number of URLs in the text."""
    def count_urls(text):
        url_pattern = r'<a\s+href=["\'](https?://[^"\']+)["\']>|https?://[^\s<>\"]+'
        return len(re.findall(url_pattern, text))

    df['nb_URLs'] = df[column].apply(count_urls)
    return df


def sentiment_df(df, column):
    """Add polarity and subjectivity scores using TextBlob."""
    def get_sentiment(text):
        blob = TextBlob(text)
        return pd.Series([blob.sentiment.polarity, blob.sentiment.subjectivity])

    df[['polarity', 'subjectivity']] = df[column].apply(get_sentiment)
    return df


def count_stopwords_df(df, column):
    """Count the number of French stopwords in the text."""
    stop_words = set(stopwords.words('french'))

    def count_stopwords(text):
        words = re.findall(r'\b\w+\b', text.lower())
        return sum(1 for word in words if word in stop_words)

    df['nb_stopwords'] = df[column].apply(count_stopwords)
    return df


def count_ner_df(df, column):
    """Count the number of named entities in the text."""
    def count_ner(text):
        doc = nlp(text)
        return len(doc.ents)

    df['nb_NER'] = df[column].apply(count_ner)
    return df


# -----------------------------------------------------------------------------
# NORMALIZATION FUNCTIONS
# -----------------------------------------------------------------------------


def normalize_df(df,
                 columns_to_normalize_by_words=("ADJ", "ADV", "NOUN", "PUNCT", "VERB", "nb_URLs", "nb_emojis"),
                 columns_to_normalize_by_len=("PUNCT", "NOUN", "VERB")):
    """Normalize selected columns by number of words or characters."""
    required_columns = set(columns_to_normalize_by_words) | set(columns_to_normalize_by_len) | {"nb_words", "nb_char"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"normalize_df: missing expected columns: {missing}")

    normalized_df = df.copy()

    def normalize_by_column(df, main_column, norm_column):
        return df[main_column] / df[norm_column]

    # Normalize by words
    for column in columns_to_normalize_by_words:
        normalized_df[column + "_norm_by_words"] = normalize_by_column(df, column, "nb_words")

    # Normalize by characters
    for column in columns_to_normalize_by_len:
        normalized_df[column + "_norm_by_char"] = normalize_by_column(df, column, "nb_char")

    return normalized_df


# -----------------------------------------------------------------------------
# MAIN ANALYSIS FUNCTION
# -----------------------------------------------------------------------------

def analyze_df(df, column='text',
               pos_tags=("ADJ", "ADV", "NOUN", "PUNCT", "VERB"),
               punct_list=(',', '.', ';', ':', '-', '(', ')', '[', ']', '{', '}', '...', '”', '“', '"', "'", '!', '?')):
    """
    Enrich a DataFrame with multiple text-based features extracted from the specified column.

    Returns:
        tuple: (df_with_counts, df_with_counts_and_normalized_values)
    """

    # Basic counts
    count_df = length_df(df.copy(), column)
    count_df = nb_words_df(count_df, column)
    count_df = count_punct_df(count_df, column, punct_list)
    count_df = count_pos_df(count_df, column, pos_tags)
    count_df = count_emojis_df(count_df, column)
    count_df = count_urls_df(count_df, column)
    count_df = sentiment_df(count_df, column)
    count_df = count_stopwords_df(count_df, column)
    count_df = count_ner_df(count_df, column)

    # Normalized features
    normalized_count_df = normalize_df(count_df)

    return count_df, normalized_count_df
