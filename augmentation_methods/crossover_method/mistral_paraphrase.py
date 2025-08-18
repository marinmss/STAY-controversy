import ollama
from ollama import Client
import pandas as pd
from tqdm import tqdm

from table import save_html_table


def paraphrase_comment(text):
  client = Client(host='http://localhost:11434')
  model = 'mistral'
  prompt = f"Paraphrase en français le commentaire suivant en gardant le sens mais en utilisant d'autres mots :\n\n'{text}'"
  response = client.generate(model=model, prompt=prompt)
  return response['response'].strip()

def paraphrase_df(df, text_column='text'):
  df_copy = df.copy()
  tqdm.pandas(desc="Paraphrasing")
  df_copy['paraphrased'] = df_copy[text_column].progress_apply(paraphrase_comment)
  return df_copy

