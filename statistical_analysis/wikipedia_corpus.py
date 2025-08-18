import wikipedia
import re
import html
import pandas as pd


wikipedia.set_lang("fr")

def get_clean_wiki_content(page_name):
    page = wikipedia.page(page_name)
    text = page.content
    
    text = html.unescape(text)

    sections_to_remove = [
        "Voir aussi",
        "Notes et références",
        "Bibliographie",
        "Articles connexes"
    ]

    for section in sections_to_remove:
        pattern = rf"==+\s*{section}\s*==+.*?(?=(\n==[^=]|$))"
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"^=+\s*.*?\s*=+$", "", text, flags=re.MULTILINE)
    text = text.strip()

    return text

wikipedia_pages = ["Autosuffisance","Autarcie","Développement durable","Décroissance durable","Écovillage","Écoconstruction",
    "Écoville","Écocitoyenneté","Haute qualité environnementale","Simplicité volontaire","Souveraineté","Système de support de vie",
    "Taux d'autosuffisance alimentaire","Vaisseau spatial"]
wiki_contents = [get_clean_wiki_content(page) for page in wikipedia_pages]

wiki_data = {'text': wiki_contents}
wiki_df = pd.DataFrame(wiki_data)
wiki_df.to_csv("/home/marina/Documents/fac/masterTAL/M2/stage/clean_code/data/dataframes/wiki_df.csv")

