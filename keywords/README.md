FR: Evaluation de mots-clés thématiques

Sortie
Chaque fichier généré contient les éléments suivants
- le titre de la thématique
- le label correspondant
- la configuration utilisée pour établir les scores décrits, contenant:
    - embedding model (par défaut, SentenceTransformer-all-MiniLM-L6-v2)
    - isolation model (par défaut, IsolationForest)
    - le nombre de commentaires STAY labelisés selon la thématique concernée
    - le nombre de mots-clés associés à la thématique
- une liste des mots-clés, suivi des scores suivants, propre à chaque mot-clé:
    - score de similarité sémantique avec l'intégralité du corpus, défini comme la valeur moyenne des similarités cosines avec les embeddings de l'intégralité des commentaires du corpus. Ce score est contenu entre 0 et 1. Une valeur proche de 0 décrit un élément en moyenne distant du corpus, tandis qu'une valeur se répprochant de 1 représente un élément proche de celui-ci.
    - score d'isolation, défini comme un score d'anomalie du mot-clé par rapport à l'ensemble des mots-clés de la thématique. Plus le score est haut, plus l'élément est anormal; les scores positifs représentent les éléments isolés, les scores négatifs les éléments typiques à l'ensemble.
- les valeurs moyennes de ces scores calculées à partir des scores de tous les mots-clés de la thématique
- deux scores propres aux mots-clés de cette thématique en tant qu'ensemble, définis comme:
    - la moyenne des similarités cosines entre tous les mots-clés de la thématique et tous les commentaires de la thématique
    - le pourcentage de commentaires de la thématique contenant au moins un mot-clé de la thématique



------------------------------------------------------------------------------------------------------------------
EN: Thematic keyword evaluation

Output
Each generated file contains the following elements:
- the title of the theme
- the corresponding label
- the configuration used to establish the reported scores, including:
    - embedding model (default: SentenceTransformer-all-MiniLM-L6-v2)
    - isolation model (default: IsolationForest)
    - the number of comments labeled within the STAY project and related to the theme
    - the number of keywords associated with the theme
- a list of the keywords, followed by the following scores for each keyword:
    - semantic similarity score with the entire corpus, defined as the average cosine similarity with the embeddings of all comments in the corpus. This score ranges from 0 to 1. A value close to 0 indicates that, on average, the element is distant from the corpus, while a value close to 1 represents an element close to it.
    - isolation score, defined as an anomaly score of the keyword compared to all the keywords of the theme. The higher the score, the more anomalous the element; positive scores represent isolated elements, while negative scores represent elements typical of the set.
- the average values of these scores, calculated from all the keywords of the theme
- two scores specific to the keywords of this theme as a whole, defined as:
    - the average cosine similarity between all the keywords of the theme and all the comments of the theme
    - the percentage of comments within the theme containing at least one keyword of the theme