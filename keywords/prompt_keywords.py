import subprocess

def ollama_gen(prompt:str) -> str:
    output = subprocess.run(
        ['ollama', 'run', 'llama3'],
        input = prompt.encode(),
        stdout = subprocess.PIPE
        )
    return output.stdout.decode()

def main():
    # prompt = """Lis attentivement la définition suivante de la thématique 'Gestion des maladies et des ravageurs (nom court: Maladies/Ravageurs):
    # "Cette classe inclut les actions mises en place pour gérer les maladies et ravageurs des plantes cultivées (arbres compris), comme les limaces, les pucerons, les maladies fongiques, etc... Peut inclure, par exemple, des actions telles que le traintement avec bouillie bordelaise, les traitements par des purins, le lâcher de poules ou de canards indiens, le choix des porte-greffe ou des variétés résistances, l'organisation spatiale des cultures pour éviter la concentration d'une même espèce, la plantation d'espèces végétales réputées protectrices."
    
    # Voici le vocabulaire graine correspondant à cette thématique:
    # "maladie, ravageur, limace, puceron, maladie fongique, traitement, bouillie bordelaise, purin, lâcher de poule, lâcher de canards indiens, porte-greffe, variété résistante, oragnisation spaciale des cultures, espèce végétale protectrice, protecteur, protectrice"
    
    # Génère 20 nouveaux mots-clés en français propres à la thématique, afin d'étendre le vocabulaire graine.

    # "Renvoie les mots-clés générés sous le format suivant:
    # - Mot proposé : [ton mot]
    # - Confiance (0 à 100%): [ton estimation]
    # - Pourquoi ce mot?: [une justification concise, en 1 phrase]
    # """

    # response = ollama_gen(prompt)
    # with open("/home/marina/stageM2/output/prompt_gen_maladies.txt", "w", encoding="utf-8") as file:
        # file.write("Prompt: \n")
        # file.write(prompt)
        # file.write("\n\n")
        # file.write("Model output: \n")
        # file.write(response)

    ##############################################################################

    # prompt = """Lis attentivement la définition suivante de la thématique 'Gestion de l'eau (paillage, irrigation, strates végétales, eau du réseau, de pluie ou de source (nom court : Eau):
    # "Cette classe inclut les actions mises en place pour gérer l'eau dans la production agricole - qu'il s'agisse d'accès à l'eau (puits, source, réseau d'eau potable, ...), de manque d'eau (sécheresse..), d'excès d'eau (sols hydromorphes..), les économies d'eau pour éviter le gaspillage et réduire les coûts. Ces actions peuvent être, par exemple, la mise en place de paillage pour éviter l'évaporation en été, l'irrigation par goutte-à-goutte, etc."
    
    # Voici le vocabulaire graine correspondant à cette thématique:
    # "accès à l'eau, puits, source, réseau d'eau potable, manque d'eau, sécheresse, excès d'eau, sol hydromorphe, hydromorphe, économie d'eau, eau, gaspillage, paillage, évaporation, irrigation, goutte, goutte-à-goutte"
    
    # Génère 20 nouveaux mots-clés en français propres à la thématique, afin d'étendre le vocabulaire graine.

    # Renvoie les mots-clés générés sous le format suivant:
    # - Mot proposé : [ton mot]
    # - Confiance (0 à 100%): [ton estimation]
    # - Pourquoi ce mot?: [une justification concise, en 1 phrase]
    # """

    # response = ollama_gen(prompt)
    # with open("/home/marina/stageM2/output/prompt_gen_eau.txt", "w", encoding="utf-8") as file:
    #     file.write("Prompt: \n")
        # file.write(prompt)
        # file.write("\n\n")
        # file.write("Model output: \n")
        # file.write(response)

    ###############################################################################

    prompt = """Lis attentivement la définition suivante de la thématique 'Gestion de l'adéquation entre qualités du sol et besoins des plantes cultivées (nom court : Sol):
    "Cette classe inclut des actions mises en place pour rendre le sol le plus en adéquation possible avec les besoins des plantes cultivées, en termes de structure, composition et fertilité. Quelques exemples de telles actions sont l'apport d'engrais organique ou pas (fumiers divers, engrais de synthèse ; apport de compost) ; l'apport de biomasse en paillage (paille, foin, broyat, brf, feuilles mortes, branchages coupés pour cet objectif) ; favoriser le vivant dans le sol ; pratiquer la perturbation des végétaux « sacrificiels » pour stimuler les végétaux autour d'eux."
    
    Voici le vocabulaire graine correspondant à cette thématique:
    "structure, composition, fertilité, engrais, engrais organique, fumiers, apport, biomasse, engrais de synthèse, compost, paille, foin, broyat, brf, feuilles mortes, branchages coupés"
    
    Génère 20 nouveaux mots-clés en français propres à la thématique, afin d'étendre le vocabulaire graine.

    Renvoie les mots-clés générés sous le format suivant:
    - Mot proposé : [ton mot]
    - Confiance (0 à 100%): [ton estimation]
    - Pourquoi ce mot?: [une justification concise, en 1 phrase]
    """
    
    response = ollama_gen(prompt)
    with open("/home/marina/stageM2/output/prompt_gen_sol.txt", "w", encoding="utf-8") as file:
        file.write("Prompt: \n")
        file.write(prompt)
        file.write("\n\n")
        file.write("Model output: \n")
        file.write(response)

    ###############################################################################

    prompt = """Lis attentivement la définition suivante de la thématique 'Gestion des adventices (nom court : Adventices):
    "Cette classe inclut des actions mises en place pour gérer les végétaux indésirables, aussi nommées « mauvaises herbes » ou adventices. Des exemples incluent l'utilisation de bâches en tissu renforcé, la mise en place de couvert végétaux, la densification des cultures, le désherbage manuel, mécanique ou chimique."
    
    Voici le vocabulaire graine correspondant à cette thématique:
    "adventice, végétaux indésirables, végétal indésirable, mauvaise herbe, bâche en tissu renforcé, couvert végétaux, densification des cultures, densification, désherbage, deshérbage manuel, désherbage mécanique, désherbage chimique, récolte"
    
    Génère 20 nouveaux mots-clés en français propres à la thématique, afin d'étendre le vocabulaire graine.

    Renvoie les mots-clés générés sous le format suivant:
    - Mot proposé : [ton mot]
    - Confiance (0 à 100%): [ton estimation]
    - Pourquoi ce mot?: [une justification concise, en 1 phrase]
    """

    response = ollama_gen(prompt)
    with open("/home/marina/stageM2/output/prompt_gen_adventices.txt", "w", encoding="utf-8") as file:
        file.write("Prompt: \n")
        file.write(prompt)
        file.write("\n\n")
        file.write("Model output: \n")
        file.write(response)

    ###############################################################################

    prompt = """Lis attentivement la définition suivante de la thématique 'Récolte (nom court : Récolte):
    "Cette classe inclut ce qui concerne les récoltes et n'est pas couvert par d'autres classes."
    
    Voici le vocabulaire graine correspondant à cette thématique:
    "récolte"
    
    Génère 20 nouveaux mots-clés en français propres à la thématique, afin d'étendre le vocabulaire graine.

    Renvoie les mots-clés générés sous le format suivant:
    - Mot proposé : [ton mot]
    - Confiance (0 à 100%): [ton estimation]
    - Pourquoi ce mot?: [une justification concise, en 1 phrase]
    """

    response = ollama_gen(prompt)
    with open("/home/marina/stageM2/output/prompt_gen_recolte.txt", "w", encoding="utf-8") as file:
        file.write("Prompt: \n")
        file.write(prompt)
        file.write("\n\n")
        file.write("Model output: \n")
        file.write(response)
    


if __name__ == '__main__':
    main()