import pandas as pd
import random
from itertools import product

from augmentation_methods.crossover_method.adverbs import MANUAL_DICT, CLEAN_LOUVAIN_DICT, CATEGORY_DICT, INTERCHANGEABLE_CATEGORIES, get_random_adverb, is_interchangeable_category
from augmentation_methods.crossover_method.split import sa_split, louvain_split


# Same Adverb ==================================================
def combine_sa(split_df, max_diff = None, adv_dict = MANUAL_DICT, claim_label=1, premise_label=1, output_label=1):
    
    expected_columns = {'text', 'label', 'topic', 'before', 'adverb', 'after'}
    if not expected_columns.issubset(split_df.columns):
        raise ValueError(f"combine_sa: missing expected columns: {expected_columns - set(split_df.columns)}")

    all_augmented = []

    for topic_label, topic_df in split_df.groupby('topic'):

        for adverb_label, adverb_df in topic_df.groupby('adverb'):

            before_df = adverb_df[(adverb_df["label"] == claim_label)].dropna(subset=["after"]).copy()
            after_df = adverb_df[(adverb_df["label"] == premise_label)].dropna(subset=["before"]).copy()

            if before_df.empty or after_df.empty:
                continue

            for (_, before_row), (_, after_row) in product(before_df.iterrows(), after_df.iterrows()):
                
                if before_row.name == after_row.name:
                    continue
                
                before = before_row["before"].strip()
                before_comment = before_row["text"].strip()
                after  = after_row["after"].strip()
                after_comment = after_row["text"].strip()
                adverb = adverb_label
                if not before or not after:
                    continue

                if max_diff is not None and abs(len(before.split()) - len(after.split())) > max_diff:
                    continue

                augmented_comment = f"{before} {adverb} {after}"

                all_augmented.append({
                    'text': augmented_comment.strip(),
                    'label': int(output_label),
                    'topic': topic_label,
                    'before': before,
                    'adverb': adverb,
                    'after': after,
                    'before_comment': before_comment,
                    'after_comment': after_comment,
                    'before_id': before_row.name,
                    'after_id': after_row.name
                })

    return pd.DataFrame(all_augmented)


def generate_sa(df, max_diff = None, adv_dict = MANUAL_DICT):
    split_df = sa_split(df, adv_dict)

    ccc_aug_df = combine_sa(split_df = split_df, adv_dict=adv_dict, claim_label=1, premise_label=1, output_label=1)
    ccc_aug_df = ccc_aug_df.drop_duplicates()
    ccc_aug_df['origin'] = 'CCC'

    cnc_aug_df = combine_sa(split_df = split_df, adv_dict=adv_dict, claim_label=1, premise_label=0, output_label=1)
    cnc_aug_df = cnc_aug_df.drop_duplicates()
    cnc_aug_df['origin'] = 'CNC'

    ncc_aug_df = combine_sa(split_df = split_df, adv_dict=adv_dict, claim_label=0, premise_label=1, output_label=1)
    ncc_aug_df = ncc_aug_df.drop_duplicates()
    ncc_aug_df['origin'] = 'NCC'

    nnn_aug_df = combine_sa(split_df = split_df, adv_dict=adv_dict, claim_label=0, premise_label=0, output_label=0)
    nnn_aug_df = nnn_aug_df.drop_duplicates()
    nnn_aug_df['origin'] = 'NNN'

    aug_df = pd.concat([ccc_aug_df, cnc_aug_df, ncc_aug_df, nnn_aug_df], ignore_index=True)
    aug_df = aug_df.sample(frac=1).reset_index(drop=True)
    aug_df = aug_df.drop_duplicates()

    print("\n")
    print("==============================================")
    print("Generated using argumentative cross-over Same Adverb: ")
    if max_diff is None:
        print("Without maximum length difference\n")
    else:
        print(f"With maximum length different set to {max_diff}\n")
    print(f"{len(ccc_aug_df)} CCC comments")
    print(f"{len(cnc_aug_df)} CNC comments")
    print(f"{len(ncc_aug_df)} NCC comments")
    print(f"{len(nnn_aug_df)} NNN comments")
    print(f"{len(ccc_aug_df)+len(cnc_aug_df)+len(ncc_aug_df)} controversial comments")
    print(f"{len(aug_df)} comments in total")
    print("==============================================")



    return aug_df


# LOUVAIN ================================================
def combine_louvain(split_df, CATEGORY_DICT = CATEGORY_DICT, claim_label=1, premise_label=1, output_label=1):
    
    expected_cols = {'text', 'label', 'topic', 'categories', 'before', 'adverb', 'after'}
    if not expected_cols.issubset(split_df.columns):
        raise ValueError(f"combine_louvain: missing expected columns: {expected_cols - set(split_df.columns)}")

    all_augmented = []

    for topic_label, group_df in split_df.groupby('topic'):
        before_df = group_df[(group_df["label"] == claim_label)].dropna(subset=["after"]).copy()
        after_df = group_df[(group_df["label"] == premise_label)].dropna(subset=["before"]).copy()

        if before_df.empty or after_df.empty:
            continue

        for (_, before_row), (_, after_row) in product(before_df.iterrows(), after_df.iterrows()):
            if before_row.name == after_row.name:
                    continue
            
            before_categories = before_row['categories']
            after_categories = after_row['categories']
            common_categories = [category for category in before_categories if category in after_categories]

            for category in common_categories:
                new_adverb = get_random_adverb(category, CATEGORY_DICT)
                is_inter_fam = is_interchangeable_category(category, INTERCHANGEABLE_CATEGORIES)

                if is_inter_fam:
                    before = before_row['text'].strip()
                    after = after_row['text'].strip()
                else:
                    before = after_row['text'].strip()
                    after = before_row['text'].strip()

                augmented_comment = f"{before} {new_adverb} {after}"

                all_augmented.append({
                    'text': augmented_comment.strip(),
                    'label': int(output_label),
                    'topic': topic_label,
                    'categories': category,
                    'before': before,
                    'adverb': new_adverb,
                    'after': after,
                    'before_id': before_row.name,
                    'after_id': after_row.name
                })

    return pd.DataFrame(all_augmented)


def generate_louvain(df, flat_adv_dict=CLEAN_LOUVAIN_DICT, CATEGORY_DICT = CATEGORY_DICT):
    split_df = louvain_split(df, flat_adv_dict)

    ccc_aug_df = combine_louvain(split_df = split_df, CATEGORY_DICT = CATEGORY_DICT, claim_label=1, premise_label=1, output_label=1)
    ccc_aug_df = ccc_aug_df.drop_duplicates()
    ccc_aug_df['origin'] = 'CCC'

    cnc_aug_df = combine_louvain(split_df = split_df, CATEGORY_DICT = CATEGORY_DICT, claim_label=1, premise_label=0, output_label=1)
    cnc_aug_df = cnc_aug_df.drop_duplicates()
    cnc_aug_df['origin'] = 'CNC'

    ncc_aug_df = combine_louvain(split_df = split_df, CATEGORY_DICT = CATEGORY_DICT, claim_label=0, premise_label=1, output_label=1)
    ncc_aug_df = ncc_aug_df.drop_duplicates()
    ncc_aug_df['origin'] = 'NCC'

    nnn_aug_df = combine_louvain(split_df = split_df, CATEGORY_DICT = CATEGORY_DICT, claim_label=0, premise_label=0, output_label=0)
    nnn_aug_df = nnn_aug_df.drop_duplicates()
    nnn_aug_df['origin'] = 'NNN'

    aug_df = pd.concat([ccc_aug_df, cnc_aug_df, ncc_aug_df , nnn_aug_df], ignore_index=True)
    aug_df = aug_df.sample(frac=1).reset_index(drop=True)
    aug_df = aug_df.drop_duplicates()

    print("\n")
    print("==============================================")
    print("Generated using argumentative cross-over Louvain: \n")
    print(f"{len(ccc_aug_df)} CCC comments")
    print(f"{len(cnc_aug_df)} CNC comments")
    print(f"{len(ncc_aug_df)} NCC comments")
    print(f"{len(nnn_aug_df)} NNN comments")
    print(f"{len(ccc_aug_df)+len(cnc_aug_df)+len(ncc_aug_df)} controversial comments")
    print(f"{len(aug_df)} comments in total")
    print("==============================================")

    return aug_df
