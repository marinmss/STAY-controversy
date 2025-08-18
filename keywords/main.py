from config import Config
from agrokeyword import Agrokeyword
from kwset import KwSet

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import IsolationForest


def main():
        df = pd.read_csv("/home/marina/stageM2/data/dataset.csv")

        config = Config(
                df = df,
                embedding_model=SentenceTransformer("all-MiniLM-L6-v2"),
                isolation_model = IsolationForest(contamination=0.1, random_state=45))

        kw_mal_list = [Agrokeyword(term = "insecte nuisible", config=config),
        Agrokeyword(term = "acarien", config=config),
        Agrokeyword(term = "nématode", config=config),
        Agrokeyword(term = "champignon", config=config),
        Agrokeyword(term = "bactérie", config=config),
        Agrokeyword(term = "virus végétal", config=config),
        Agrokeyword(term = "lutte biologique", config=config),
        Agrokeyword(term = "piège à phéromones", config=config),
        Agrokeyword(term = "rotation des cultures", config=config),
        Agrokeyword(term = "désinfection", config=config),
        Agrokeyword(term = "résistance variétale", config=config),
        Agrokeyword(term = "biopesticide", config=config),
        Agrokeyword(term = "traitement préventif", config=config),
        Agrokeyword(term = "traitement curatif", config=config),
        Agrokeyword(term = "parasite", config=config),
        Agrokeyword(term = "larve", config=config),
        Agrokeyword(term = "mildiou", config=config),
        Agrokeyword(term = "oïdum", config=config),
        Agrokeyword(term = "tavelure", config=config),
        Agrokeyword(term = "puceron noir", config=config)]

        KWmal = KwSet(name = "Maladies et ravageurs", kw_list = kw_mal_list, label = "P2", config=config)
        KWmal.compute_keyword_score()
        KWmal.compute_kwset_score()
        print(KWmal)




        kw_eau_list = [Agrokeyword(term = "irrigation localisée", config=config),
        Agrokeyword(term = "récupération d'eau de pluie", config=config),
        Agrokeyword(term = "gestion durable de l'eau", config=config),
        Agrokeyword(term = "drainage", config=config),
        Agrokeyword(term = "humidité du sol", config=config),
        Agrokeyword(term = "capacité de rétention", config=config),
        Agrokeyword(term = "micro-irrigation", config=config),
        Agrokeyword(term = "arrosage automatique", config=config),
        Agrokeyword(term = "réservoir d'eau", config=config),
        Agrokeyword(term = "gestion des eaux pluviales", config=config),
        Agrokeyword(term = "sécheresse prolongée", config=config),
        Agrokeyword(term = "stress hydrique", config=config),
        Agrokeyword(term = "pompage", config=config),
        Agrokeyword(term = "canal d'irrigation", config=config),
        Agrokeyword(term = "bassin de rétention", config=config),
        Agrokeyword(term = "évapotranspiration", config=config),
        Agrokeyword(term = "qualité de l'eau", config=config),
        Agrokeyword(term = "eau souterraine", config=config),
        Agrokeyword(term = "eau de surface", config=config),
        Agrokeyword(term = "gestion intégrée de l'eau", config=config)]

        KWeau = KwSet(name = "Eau", kw_list = kw_eau_list, label = "P3", config=config)
        KWeau.compute_keyword_score()
        KWeau.compute_kwset_score()
        print(KWeau)





        kw_sol_list = [Agrokeyword(term = "pH du sol", config=config),
        Agrokeyword(term = "texture du sol", config=config),
        Agrokeyword(term = "aération du sol", config=config),
        Agrokeyword(term = "micro-organismes du sol", config=config),
        Agrokeyword(term = "mycorhizes", config=config),
        Agrokeyword(term = "fertilisation", config=config),
        Agrokeyword(term = "amendement", config=config),
        Agrokeyword(term = "rotation culturale", config=config),
        Agrokeyword(term = "culture intercalaire", config=config),
        Agrokeyword(term = "compostage", config=config),
        Agrokeyword(term = "engrais vert", config=config),
        Agrokeyword(term = "labour", config=config),
        Agrokeyword(term = "non-labour", config=config),
        Agrokeyword(term = "structure granulaire", config=config),
        Agrokeyword(term = "érosion", config=config),
        Agrokeyword(term = "biodiversité du sol", config=config),
        Agrokeyword(term = "capacité d'échange cationique", config=config),
        Agrokeyword(term = "matière organique", config=config),
        Agrokeyword(term = "minéralisation", config=config),
        Agrokeyword(term = "bioturbation", config=config)]

        KWsol = KwSet(name = "Sol", kw_list = kw_sol_list, label = "P6", config=config)
        KWsol.compute_keyword_score()
        KWsol.compute_kwset_score()
        print(KWsol)




        kw_adv_list = [Agrokeyword(term = "désherbage thermique", config=config),
        Agrokeyword(term = "désherbage chimique séléctif", config=config),
        Agrokeyword(term = "désherbage manuel intensif", config=config),
        Agrokeyword(term = "désherbage mécanique rotatif", config=config),
        Agrokeyword(term = "paillage plastique", config=config),
        Agrokeyword(term = "paillage organique", config=config),
        Agrokeyword(term = "couverts végétaux d'interculture", config=config),
        Agrokeyword(term = "compétition culturale", config=config),
        Agrokeyword(term = "semis direct", config=config),
        Agrokeyword(term = "binage", config=config),
        Agrokeyword(term = "sarclage", config=config),
        Agrokeyword(term = "désherbage ciblé", config=config),
        Agrokeyword(term = "herbicide", config=config),
        Agrokeyword(term = "herbicide naturel", config=config),
        Agrokeyword(term = "gestion intégrée des adventices", config=config),
        Agrokeyword(term = "barrière physique", config=config),
        Agrokeyword(term = "désherbage écologique", config=config),
        Agrokeyword(term = "désherbage biologique", config=config),
        Agrokeyword(term = "désherbage par solarisation", config=config),
        Agrokeyword(term = "désherbage par mulching", config=config)]

        KWadv = KwSet(name = "Adventices", kw_list = kw_adv_list, label = "P7", config=config)
        KWadv.compute_keyword_score()
        KWadv.compute_kwset_score()
        print(KWadv)





        kw_rec_list = [Agrokeyword(term = "matériel de récolte", config=config),
        Agrokeyword(term = "récolte mécanisée", config=config),
        Agrokeyword(term = "récolte manuelle", config=config),
        Agrokeyword(term = "calendrier de récolte", config=config),
        Agrokeyword(term = "maturité physiologique", config=config),
        Agrokeyword(term = "stockage en silo", config=config),
        Agrokeyword(term = "conditionnement", config=config),
        Agrokeyword(term = "triage", config=config),
        Agrokeyword(term = "nettoyage post-récolte", config=config),
        Agrokeyword(term = "transport des récoltes", config=config),
        Agrokeyword(term = "récolte précoce", config=config),
        Agrokeyword(term = "récolte tardive", config=config),
        Agrokeyword(term = "récolte en vert", config=config),
        Agrokeyword(term = "récolte en sec", config=config),
        Agrokeyword(term = "récolte en grappes", config=config),
        Agrokeyword(term = "récoltes en bottes", config=config),
        Agrokeyword(term = "récolte sous serre", config=config),
        Agrokeyword(term = "récolte de précision", config=config),
        Agrokeyword(term = "récolte automatisée", config=config),
        Agrokeyword(term = "récolte manuelle sélective", config=config)]

        KWrec = KwSet(name = "Récoltes", kw_list = kw_rec_list, label = "PRecolte", config=config)
        KWrec.compute_keyword_score()
        KWrec.compute_kwset_score()
        print(KWrec)

        kwsets = [KWmal, KWeau, KWsol, KWadv, KWrec]
        with open("/home/marina/stageM2/output/zeroshot_keyword_scores.txt", "w", encoding="utf-8") as f:
                for kwset in kwsets:
                       f.write(str(kwset))
                       f.write("\n")
        

































        mal_list = ["lutte chimique","lutte culturale","biocontrôle","insectes auxiliaires","résistance génétique","piégeage lumineux","rotation des parcelles",
                    "désinfection des outils","traitement préventif","surveillance phytosanitaire","seuil de nuisibilité","mildiou","oïdium","pucerons ailés",
                    "larves","nuisibilité","parasitoïdes","champignons pathogènes","virus transmis par insectes","traitement curatif"]
        kw_mal_list = [Agrokeyword(term = t, config=config) for t in mal_list]

        KWmal = KwSet(name = "Maladies et ravageurs", kw_list = kw_mal_list, label = "P2", config=config)
        KWmal.compute_keyword_score()
        KWmal.compute_kwset_score()
        print(KWmal)



        eau_list = ["irrigation localisée","récupération d'eau de pluie","gestion durable de l'eau","capteurs d'humidité","irrigation automatisée",
                    "gestion des eaux souterraines","zones humides","réduction de l'évaporation","irrigation par aspersion","gestion des crues",
                    "réutilisation des eaux usées","bassin d'infiltration","gestion des nappes phréatiques","irrigation de précision","gestion des sols salins",
                    "stress hydrique chronique","irrigation intermittente","gestion des eaux de drainage","microclimat","réduction des pertes d'eau"]
        kw_eau_list = [Agrokeyword(term = t, config=config) for t in eau_list]

        KWeau = KwSet(name = "Eau", kw_list = kw_eau_list, label = "P3", config=config)
        KWeau.compute_keyword_score()
        KWeau.compute_kwset_score()
        print(KWeau)






        sol_list = ["analyse chimique du sol","amendement calcaire","fixation du carbone","biodégradation","structure granulométrique","compostage",
                    "rotation des cultures","culture intercalaire","labour superficiel","non-labour","fertilisation organique","fertilisation minérale",
                    "microbiote du sol","mycorhizes arbusculaires","érosion hydrique","érosion éolienne","compactage du sol","drainage du sol",
                    "couverture du sol","stabilisation du sol"]
        kw_sol_list = [Agrokeyword(term = t, config=config) for t in sol_list]

        KWsol = KwSet(name = "Sol", kw_list = kw_sol_list, label = "P6", config=config)
        KWsol.compute_keyword_score()
        KWsol.compute_kwset_score()
        print(KWsol)



        adv_list = ["désherbage thermique","désherbage manuel","désherbage chimique sélectif","désherbage mécanique rotatif","paillage organique",
                    "paillage plastique biodégradable","couverts végétaux d'interculture","compétition culturale","semis direct","binage","sarclage",
                    "désherbage ciblé","herbicide systémique","herbicide de contact","désherbage par solarisation","désherbage par flammes","désherbage par eau chaude",
                    "gestion intégrée des adventices","rotation des cultures","densification des cultures"]
        kw_adv_list = [Agrokeyword(term = t, config=config) for t in adv_list]

        KWadv = KwSet(name = "Adventices", kw_list = kw_adv_list, label = "P7", config=config)
        KWadv.compute_keyword_score()
        KWadv.compute_kwset_score()
        print(KWadv)

        rec_list = ["récolte manuelle","récolte mécanisée","calendrier de récolte","stockage en silo","conditionnement","transport post-récolte",
                    "triage","sélection des fruits","récolte précoce","récolte tardive","récolte en plusieurs passages","récolte sous serre",
                    "récolte de précision","récolte automatisée","récolte par secouage","récolte par coupe","récolte par cueillette","récolte en grappes",
                    "récolte en vrac","récolte et stockage à froid"]
        kw_rec_list = [Agrokeyword(term = t, config=config) for t in rec_list]

        KWrec = KwSet(name = "Récoltes", kw_list = kw_rec_list, label = "PRecolte", config=config)
        KWrec.compute_keyword_score()
        KWrec.compute_kwset_score()
        print(KWrec)

        kwsets = [KWmal, KWeau, KWsol, KWadv, KWrec]
        with open("/home/marina/stageM2/output/fewshots_keyword_scores.txt", "w", encoding="utf-8") as f:
                for kwset in kwsets:
                       f.write(str(kwset))
                       f.write("\n")

if __name__ == "__main__":
    main()