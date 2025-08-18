import pandas as pd
import requests
from bs4 import BeautifulSoup

def get_text(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() 
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        content = "\n".join([p.get_text(strip=True) for p in paragraphs])
        return content
    except Exception as e:
        print(f"Erreur sur {url}: {e}")
        return None

def get_text_file_content(file):
    try:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        print(f"Error: file '{file}' not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

urls = [
    "https://jardinage-permaculture.fr/definition-de-la-permaculture",
    "https://jardinage-permaculture.fr/blog/guide-permaculture/faut-il-enlever-les-gourmands-des-plants-de-tomates-en-permaculture",
    "https://jardinage-permaculture.fr/blog/guide-permaculture/quels-animaux-vivent-dans-le-compost",
    "https://jardinage-permaculture.fr/blog/guide-permaculture/comment-faire-son-compost-comprendre-le-compostage-de-dechets-organiques",
    "https://jardinage-permaculture.fr/blog/guide-permaculture/creer-une-butte-en-lasagne"
]

text_urls = [get_text(url) for url in urls]

blog1 = get_text_file_content("/home/marina/Documents/fac/masterTAL/M2/stage/clean_code/data/blogs/blog_content1.txt")

if blog1 is not None:
    text_urls = [blog1] + text_urls

text_urls = [t for t in text_urls if t]
blog_df = pd.DataFrame({'text': text_urls})
blog_df.to_csv("/home/marina/Documents/fac/masterTAL/M2/stage/clean_code/data/dataframes/blog_df.csv")

