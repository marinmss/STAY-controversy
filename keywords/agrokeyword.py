from sklearn.metrics.pairwise import cosine_similarity
from config import Config

class Agrokeyword:
    def __init__(self, term, config: Config):
        
        self.embedding_model = config.embedding_model
        self.corpus = config.corpus
        self.corpus_embeddings = config.corpus_embeddings       

        self.term = term.lower()
        self.embedding = self.embedding_model.encode(term)

        self.corpus_similarity_score = None
        self.isolation_score = None
        self.keyword_score = None

    def compute_keyword_whole_corpus_average_similarity(self):
        similarities = cosine_similarity([self.embedding], self.corpus_embeddings)
        self.corpus_similarity_score = float(similarities.mean())

    def __str__(self):
        return (
        f"{self.term:<35}"
        f" | corpus similarity score: {self.corpus_similarity_score:>8.4f}"
        f" | isolation score: {self.isolation_score:>8.4f}"
    )
        


    __repr__ = __str__


    