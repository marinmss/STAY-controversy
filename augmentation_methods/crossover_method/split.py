import pandas as pd
import re

from augmentation_methods.crossover_method.adverbs import MANUAL_DICT, CLEAN_LOUVAIN_DICT

def sa_split(df, adv_dict = MANUAL_DICT):
    split_df = df.copy()

    def split_by_adverb(text, adv_dict):
        pattern = re.compile(
            r'\b(' + '|'.join(map(re.escape, adv_dict)) + r')\b',
            flags=re.IGNORECASE
        )
        match = pattern.search(text)
        if match:
            adverb = match.group(1)
            start, end = match.span()
            before = text[:start].strip()
            after  = text[end:].strip()
            return pd.Series([before, adverb, after])

        return pd.Series([None, None, None])


    split_df[['before', 'adverb', 'after']] = (split_df['text'].apply(split_by_adverb, adv_dict = adv_dict))
    split_df = split_df.dropna(subset=["before", "adverb", "after"])

    return split_df





def louvain_split(df, flat_adv_dict = CLEAN_LOUVAIN_DICT):
    split_df = df.copy()


    def split_by_adverb(text, flat_adv_dict):
        adverbs = flat_adv_dict.keys()
        pattern = re.compile(
            r'\b(' + '|'.join(map(re.escape, adverbs)) + r')\b',
            flags=re.IGNORECASE
        )
        match = pattern.search(text)
        if match:
            adverb = match.group(1)              
            families = flat_adv_dict[adverb.lower()]
            start, end = match.span()
            before = text[:start].strip()
            after  = text[end:].strip()
            return pd.Series([families, before, adverb, after])

        return pd.Series([None, None, None, None])

    split_df[['families', 'before', 'adverb', 'after']] = (split_df['text'].apply(split_by_adverb, flat_adv_dict=flat_adv_dict))
    split_df = split_df.dropna(subset=["before", "adverb", "after"])

    return split_df
