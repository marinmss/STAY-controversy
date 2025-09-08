from collections import Counter
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import nltk
import spacy

from nltk.corpus import stopwords

nlp = spacy.load("fr_core_news_sm")
nltk.download('stopwords')


def save_counter_to_file(counter, title = "Most frequent ngrams", filename="output/freq_words.txt"):
    counter = dict(counter)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(title)
        f.write("-" * 20 + "\n")
        for elt in counter:
            f.write(f"{elt} : {counter[elt]}\n")


def process_ngrams(text, ngram_size = 1, n=50, with_stopwords = False):
    words = re.findall(r'\b\w+\b', text.lower())
    if not with_stopwords:
        stopwords_list = set(stopwords.words('french'))
        words = [word for word in words if word not in stopwords_list]
    
    if ngram_size == 1:
        tokens = words
    else:
        tokens = ['_'.join(words[i:i+ngram_size]) for i in range(len(words)-ngram_size+1)]
    word_counts = Counter(tokens)
    return word_counts.most_common(n)



def get_most_frequent_ngrams(contr_text, non_contr_text, n_gram_size = 1, n=100, output_path="output/ngrams_freq/"):
    variants = [
        ("contr", contr_text),
        ("non_contr", non_contr_text),
    ]
    for label, text in variants:
        for clean in [True, False]:
            suffix = "clean" if not clean else "stopwords"
            filename = f"{label}_freq_{n_gram_size}gram_{suffix}.txt"
            counter = process_ngrams(text, n_gram_size, n, clean)
            save_counter_to_file(counter, f"{n} most frequent {n_gram_size}gram", output_path + filename)


def process_pos(text, pos_tag, n=50):
    doc = nlp(text)
    elts = [token.lemma_.lower() for token in doc if token.pos_ == pos_tag]
    freqs = Counter(elts)
    return freqs.most_common(n)


def get_most_frequent_pos(contr_text, non_contr_text, pos_tag, n=100, output_path="output/pos_freq/"):
    variants = [
        ("contr", contr_text),
        ("non_contr", non_contr_text),
    ]
    for label, text in variants:
        filename = f"{label}_freq_{pos_tag}.txt"
        counter = get_most_frequent_pos(text, pos_tag, n)
        save_counter_to_file(counter, f"{n} most frequent {pos_tag}", output_path + filename)