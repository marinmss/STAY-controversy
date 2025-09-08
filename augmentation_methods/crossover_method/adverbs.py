import random
import pandas as pd
import re
import numpy as np

from collections import defaultdict


# adverb list for Same Adverb approach
MANUAL_DICT = ["car","parce que", "en effet", "en raison de", 
                "malgré", "pour des raisons de", "cependant", 
                "néanmoins","alors", "ainsi", "aussi", "d'ailleurs", 
                "en fait", "en effet", "de surcroît", "de même", 
                "également", "puis", "ensuite", "de plus", "en outre", 
                "par exemple", "comme", "ainsi", "c'est ainsi que", 
                "c'est le cas de", "notamment", "entre autres", 
                "en particulier", "à l'image de", "comme l'illustre", 
                "comme le souligne", "tel que", "d'abord", "tout d'abord", 
                "de prime abord", "en premier lieu", "premièrement", 
                "en deuxième lieu", "en second lieu", "deuxièmement", 
                "après", "ensuite", "de plus", "quant à", "en troisième lieu", 
                "puis", "en dernier lieu", "pour conclure", "enfin", "de plus", 
                "puis", "en outre", "non seulement", "mais encore", "de surcroît", 
                "ainsi que", "également"]

# manually cleaned adverb dictionary for Louvain approach
LOUVAIN_DICT = {
    "addition": [
        "Et",
        "De plus",
        "Puis",
        "En outre"
    ],
    "alternative": [
        "Ou",
        "Ou bien",
    ],
    "but": [
        "Afin que",
        "Pour que",
        "De peur que",
        "En vue de",
        "De façon à ce que"
    ],
    "cause": [
        "Car",
        "En effet",
        "Effectivement",
        "Par",
        "Parce que",
        "Puisque",
        "Attendu que",
        "Vu que",
        "Etant donné que",
        "Grâce à",
        "Par suite de",
        "Eu égard à",
        "En raison de",
        "Du fait que",
        "Dans la mesure où",
        "Sous prétexte que"
    ],
    "comparaison": [
        "De même que",
        "Ainsi que",
        "Autant que",
        "De la même façon que",
        "Semblablement",
        "Pareillement",
        "Plus que",
        "Moins que",
        "Non moins que",
        "Selon que",
        "Suivant que",
        "Comme si"
    ],
    "concession": [
        "Quel que soit",
        "Malgré",
        "En dépit de",
        "Quoique",
        "Bien que",
        "Alors que",
        "Quel que soit",
        "Même si",
        "Ce n'est pas que",
        "Certes",
        "Bien sûr",
        "Évidemment",
        "Il est vrai que",
        "Toutefois"
    ],
    "conclusion": [
        "En conclusion",
        "Pour conclure",
        "En guise de conclusion",
        "En somme",
        "Bref",
        "Ainsi",
        "Donc",
        "En résumé",
        "En un mot",
        "Par conséquent",
        "Finalement",
        "Enfin",
        "En définitive"
    ],
    "condition": [
        "Si",
        "Au cas où",
        "A condition que",
        "Pourvu que",
        "A moins que",
        "En admettant que",
        "Pour peu que",
        "A supposer que",
        "En supposant que",
        "Dans l'hypothèse où",
        "Dans le cas où",
        "Probablement",
        "Sans doute",
        "Apparemment"
    ],
    "conséquence": [
        "Donc",
        "Aussi",
        "Partant",
        "Alors",
        "Ainsi",
        "Par conséquent",
        "si bien que",
        "D'où",
        "En conséquence",
        "Conséquemment",
        "Par suite",
        "C'est pourquoi",
        "De sorte que",
        "En sorte que",
        "De façon que",
        "De manière que",
        "Si bien que",
        "Tant et"
    ],
    "classification": [
        "D'abord",
        "Tout d'abord",
        "En premier lieu",
        "Premièrement",
        "En deuxième lieu",
        "Deuxièmement",
        "Après",
        "Ensuite",
        "De plus",
        "Quant à",
        "En troisième lieu",
        "Puis",
        "En dernier lieu",
        "Pour conclure",
        "Enfin"
    ],
    "explication": [
        "Savoir",
        "A savoir",
        "C'est-à-dire",
        "Soit"
    ],
    "illustration": [
        "Par exemple",
        "Ainsi",
        "C'est ainsi que",
        "C'est le cas de",
        "Notamment",
        "Entre autre",
        "En particulier"
    ],
    "justification": [
        "Car",
        "C'est-à-dire",
        "En effet",
        "Parce que",
        "Puisque",
        "En sorte que",
        "Ainsi",
        "C'est ainsi que",
        "Non seulement… mais encore",
        "Du fait de"
    ],
    "liaison": [
        "Alors",
        "Ainsi",
        "Aussi",
        "D'ailleurs",
        "En fait",
        "En effet",
        "De surcroît",
        "De même",
        "Également",
        "Puis",
        "Ensuite"
    ],
    "opposition": [
        "Mais",
        "Cependant",
        "Or",
        "En revanche",
        "Alors que",
        "Pourtant",
        "Par contre",
        "Tandis que",
        "Néanmoins",
        "Au contraire",
        "Pour sa part",
        "D'un autre côté",
        "En dépit de",
        "Malgré",
        "Au lieu de"
    ],
    "restriction": [
        "Cependant",
        "Toutefois",
        "Néanmoins",
        "Pourtant",
        "Mis à part",
        "En dehors de",
        "Hormis",
        "A défaut de",
        "Excepté",
        "Sauf",
        "Uniquement",
        "Simplement"
    ],
    "temps": [
        "Quand",
        "Lorsque",
        "Avant que",
        "Après que",
        "Alors que",
        "Dès lors que",
        "Tandis que",
        "Depuis que",
        "En même temps que",
        "Pendant que",
        "Au moment où"
    ]
}

# list of categories of adverbs which define substituable argumentative components
INTERCHANGEABLE_CATEGORIES = [
    "alternative",
    "comparaison",
    "classification",
    "explication",
    "illustration"
]



def flatten_adv_dict(adv_dict:list = LOUVAIN_DICT):
    """
    Flattens a given adverb dictionary by mapping each adverb to the list of categories it belongs to.

    Args:
        adv_dict (dict): python dictionary with categories as keys and adverbs as values.

    Return:
        dict: python dictionary with adverbs as keys and categories as values.
    """

    # defining dictionary to be filled in and returned
    flat_dict = defaultdict(list)

    # looping through the elements of the initial dictionary
    for category, adverbs in adv_dict.items():

        # looping through the adverbs of a given category
        for adv in adverbs:

            # lowercasing the adverb and category
            adv_lc = adv.lower()
            category_lc = category.lower()

            # storing values in the new dictionary with the adverb as the key and the category as the value
            if category_lc not in flat_dict[adv_lc]:
                flat_dict[adv_lc].append(category_lc)
    
    # returning the flattened adverb dictionary
    return flat_dict





def get_adv_positions(adverbs, df, text_column = 'text'):
    adverb_positions = {adv: [] for adv in adverbs}
    for adverb in adverbs:
        pattern = re.compile(r'\b' + re.escape(adverb) + r'\b', flags=re.IGNORECASE)
        for idx, text in df[text_column].items():
            matches = list(pattern.finditer(text))
            if not matches:
                continue

            words = text.split()
            nb_words = len(words)
            if nb_words == 0:
                continue

            for match in matches:
                char_start = match.start()
                word_index = None
                running_char_count = 0
                for i, word in enumerate(words):
                    running_char_count += len(word) + 1
                    if running_char_count > char_start:
                        word_index = i
                        break
                if word_index is not None:
                    rel_pos = (word_index + 1) / nb_words    
                    adverb_positions[adverb.lower()].append(round(rel_pos, 4))
    return adverb_positions

def get_adv_stats(adverb_positions):
    adverb_stats = []
    for adv, positions in adverb_positions.items():
        if positions:
            count = len(positions)
            mean_pos = round(np.mean(positions), 4)
            std_pos = round(np.std(positions), 4)
            median_pos = round(np.median(positions), 4)
            q1 = round(np.percentile(positions, 25), 4)
            q3 = round(np.percentile(positions, 75), 4)
        else:
            count = 0
            mean_pos = std_pos = median_pos = q1 = q3 = None
        adverb_stats.append({
            "adverb": adv,
            "count": count,
            "mean_position": mean_pos,
            "std_dev": std_pos,
            "median": median_pos,
            "Q1": q1,
            "Q3": q3
        })
    return adverb_stats

def get_middle_advs(adverb_stats):
    stats_df = pd.DataFrame(adverb_stats).sort_values("count", ascending=False)
    stats_df = stats_df[stats_df["count"] > 0].reset_index(drop=True)

    if not stats_df.empty:
        Q1 = stats_df["mean_position"].quantile(0.25)
        Q3 = stats_df["mean_position"].quantile(0.75)

        middle_adverbs_df = stats_df[
            (stats_df["mean_position"] >= Q1) &
            (stats_df["mean_position"] <= Q3)
        ].copy()

        middle_adverbs = middle_adverbs_df["adverb"].tolist()

    return middle_adverbs


def get_clean_adv_dicts(df, adv_dict=LOUVAIN_DICT, text_column='text'):
    flat_dict = flatten_adv_dict(adv_dict) 
    adverbs = flat_dict.keys()

    adverb_positions = get_adv_positions(adverbs, df, text_column)
    adverb_stats = get_adv_stats(adverb_positions)
    middle_adverbs = get_middle_advs(adverb_stats)
    middle_adverbs_set = set(middle_adverbs) 

    flat_adv_dict = {
        adv.lower(): [category.lower() for category in flat_dict[adv]]
        for adv in middle_adverbs
    }

    category_dict = {
        category.lower(): [adv.lower() for adv in advs if adv.lower() in middle_adverbs_set]
        for category, advs in adv_dict.items()
    }

    return flat_adv_dict, category_dict


def get_random_adverb(category, category_dict):
    category = category.lower()
    if category in category_dict.keys():
        return random.choice(category_dict[category])
    else:
        raise ValueError(f"category '{category}' not in dictionary")

def is_interchangeable_category(category, interchangeable_categories = INTERCHANGEABLE_CATEGORIES):
    return category in interchangeable_categories


df = pd.read_csv("./data/dataframes/clean_dataset.csv")
CLEAN_LOUVAIN_DICT, CATEGORY_DICT = get_clean_adv_dicts(df, LOUVAIN_DICT)