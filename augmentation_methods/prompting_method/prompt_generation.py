import ollama
from ollama import Client
import pandas as pd
from tqdm import tqdm
import re

def clean_prompted_df(df, text_column='text'):
    clean_df = df.copy()
    clean_df[text_column] = df[text_column].apply(lambda x: re.sub(r'^"{3}|"{3}$', '', x).strip().strip('"').strip("'"))
    clean_df = clean_df[[text_column]]
    return clean_df

def generate_zero_shot(nb_batches):
    client = Client(host='http://localhost:11434')

    zero_shot_comments = []

    for _ in tqdm(range(nb_batches), desc="Comment prompt generation"):
        try:
            response = client.generate(
                model='mistral',
                prompt=f"Génère exactement 5 commentaires YouTube controversés sur une vidéo en français portant sur l'auto-suffisance, "
                "sans utiliser le mot 'auto-suffisance'. "
                "Retourne les commentaires sous forme d'une liste numérotée, un par ligne, comme ceci :\n"
                "1. ...\n2. ...\n3. ...\n4. ...\n5. ..."
                )
            raw_text = response.get('response', '').strip()
        except Exception as e:
            print(f"Error during generation: {e}")
            continue

        raw_text = response['response'].strip()
        comments = re.findall(r'\d\.\s*(.+)', raw_text)

        if len(comments) == 5:
            zero_shot_comments.extend(comments)
        else:
            print("Formatting error. Ignored text:\n", raw_text)

    zero_shot_df = pd.DataFrame(zero_shot_comments, columns=["text"])
    clean_df = clean_prompted_df(zero_shot_df)

    print("\n")
    print("==============================================")
    print("Generated using Msistral prompting: \n")
    print(f"{len(clean_df)} comments in total")
    print("==============================================")

    return clean_df



