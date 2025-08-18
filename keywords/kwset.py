from config import Config
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

import spacy
nlp = spacy.load("fr_core_news_sm")

class KwSet:
    def __init__(self, name, kw_list, label, config:Config):

        self.name = name
        self.label = label

        self.embedding_model = config.embedding_model
        self.isolation_model = config.isolation_model
        
        self.keywords = kw_list
        self.terms = [kw.term for kw in self.keywords]
        self.term_embeddings = self.embedding_model.encode(self.terms)
        self.isolation_model.fit(self.term_embeddings)

        df = config.df
        comments = df.loc[df['topic']==label]
        self.comments = list(comments['text'])
        self.comment_embeddings = self.embedding_model.encode(self.comments)

        self.corpus_similarity = None
        self.coverage_ratio = None
    
    def load(keywords:list):
        pass

    def get_terms(self) -> set[str]:
        return set(kw.term for kw in self.keywords)

    def compute_isolation_score(self):
        isolation_model = self.isolation_model
        for kw in self.keywords:
            raw_score = isolation_model.decision_function(kw.embedding.reshape(1,-1))
            kw.isolation_score = float(-raw_score)

    def compute_keyword_score(self):
        for kw in self.keywords:
            if kw.corpus_similarity_score is None :
                kw.compute_keyword_whole_corpus_average_similarity() 
                
            if kw.isolation_score is None:
                self.compute_isolation_score() 


    def compute_corpus_similarity(self):
        term_emb = np.asarray(self.term_embeddings)      
        com_emb  = np.asarray(self.comment_embeddings)
        similarity_matrix = cosine_similarity(term_emb, com_emb)
        self.corpus_similarity = similarity_matrix.mean()

    def compute_coverage_ratio(self):
        term_lemmas = set()
        for t in self.terms:
            doc = nlp(t)
            for token in doc:
                if not token.is_stop and not token.is_punct:
                    term_lemmas.add(token.lemma_)

        covered = 0
        total = len(self.comments)

        for c in self.comments:
            doc = nlp(c)
            comment_lemmas = set(token.lemma_ for token in doc if not token.is_stop and not token.is_punct)

            if term_lemmas & comment_lemmas:
                covered += 1

        coverage_ratio = covered / total if total > 0 else 0.0
        self.coverage_ratio = coverage_ratio*100
        return self.coverage_ratio
    
    def compute_kwset_score(self):
        if self.corpus_similarity is None :
            self.compute_corpus_similarity()
                
        if self.coverage_ratio is None:
            self.compute_coverage_ratio()


    def __str__(self):
        lines = []
        lines.append(f"TOPIC: {self.name} ({self.label})\n")
        lines.append("Configuration")
        lines.append(f"- Embedding model: {type(self.embedding_model).__name__}")
        lines.append(f"- Isolation model: {type(self.isolation_model).__name__}")
        lines.append(f"- Number of comments : {len(self.comments)}")
        lines.append(f"- Number of keywords: {len(self.keywords)}\n")
        
        count = len(self.keywords)

        total_sim = 0
        total_iso = 0
        total_score = 0

        for kw in self.keywords:
            lines.append(str(kw))
            total_sim += kw.corpus_similarity_score or 0
            total_iso += kw.isolation_score or 0
            total_score += kw.keyword_score or 0

        if count > 0:
            lines.append("\nScore averages for class (over whole corpus): ")
            lines.append(f"- Mean corpus similarity score      : {total_sim / count:.4f}")
            lines.append(f"- Mean isolation score              : {total_iso / count:.4f}")

        lines.append("\nClass scores (over thematic corpus): ")
        lines.append(f"- Mean corpus similarity score      : {self.corpus_similarity}")
        lines.append(f"- Coverage ratio                    : {self.coverage_ratio} %")
        lines.append("\n\n\n\n\n")

        return "\n".join(lines)


