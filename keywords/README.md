Thematic keyword evaluation/Evaluation de mots-clés thématiques

Output/Sortie
FR:
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