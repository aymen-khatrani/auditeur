# Auditeur de traçabilité et de couverture des exigences — PoC

Besoin couvert (fiche 1.2, P1) : comparer plusieurs extractions de référentiels DOORS
(`.CUS`, `.RS`, `.AD`, `.RSTPS`, `.EL`), produire une **matrice des liens possibles**
entre niveaux (`.CUS → RSL0 → RSL1 → TPS`), **proposer un ou plusieurs requis à chaîner
avec un taux de confiance**, et **lister les écarts** quand une partie de l'exigence ne
peut pas être couverte.

Contrainte industrielle : **aucune donnée ne sort du poste**. Python + Ollama en local
(`127.0.0.1`), pas d'API externe, cache disque pour la reproductibilité.

## 1. Principe

```
extractions .xlsx DOORS
      │
      ├─ parsing générique (Cover / MetaData / ModuleRich / ViewRich)   → rtm/doors.py
      │    • niveau déduit du nom de module ; sections reconstruites
      │    • liens existants décodés (~ObjectLinks, Parent/ChildRequirements)
      │
      ├─ vivier de candidats  = 0,5 × embeddings locaux + 0,5 × TF-IDF   → rtm/match.py
      │
      ├─ score de fusion explicable (6 signaux)                          → rtm/features.py
      │    embeddings, position relative, TF-IDF, lexical, valeurs
      │    chiffrées + unités, normes citées, proximité de section
      │
      ├─ taux de confiance calibré sur les liens DOORS déjà saisis       → rtm/analysis.py
      │
      ├─ juge LLM local : valide la chaîne et isole les aspects          → rtm/judge.py
      │    non couverts (découpage de l'exigence en aspects vérifiables)
      │
      └─ livrables : matrice Excel + synthèse Markdown                   → rtm/report.py
```

Point clé : le LLM **n'invente pas de lien**. Il arbitre et explique des candidats
produits par un moteur déterministe, et sa sortie est contrainte (JSON, étiquettes
`C1..Ck`, verdicts fermés) puis filtrée. Confiance finale = `0,55 × score calibré
+ 0,45 × (verdict × confiance LLM)`.

## 2. Installation

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install openpyxl pandas scikit-learn rapidfuzz requests pyyaml pytest
ollama pull nomic-embed-text          # embeddings (274 Mo)
ollama pull qwen2.5:7b-instruct       # juge recommandé (rapide)
```

Les modèles sont déclarés dans `config.yaml` (`ollama.embed_model`, `ollama.judge_model`).
Si Ollama est injoignable ou un modèle absent, l'outil **bascule automatiquement en mode
déterministe** (TF-IDF + signaux lexicaux) au lieu d'échouer.

## 3. Utilisation

```bash
# 1. périmètre détecté : niveaux, préfixes, nombre d'exigences, liens existants
python -m rtm.cli inspect --data-dir /chemin/vers/extractions

# 2. calibration du taux de confiance sur les liens déjà saisis dans DOORS
python -m rtm.cli calibrate --data-dir /chemin/vers/extractions --all-chains

# 3. audit + livrables
python -m rtm.cli audit --data-dir /chemin/vers/extractions --all-chains \
       --judge-policy targeted --judge-budget 150 --out out

# variantes utiles
python -m rtm.cli audit --data-dir .. --chain CUS:RS --no-llm      # rapide, sans LLM
python -m rtm.cli audit --data-dir .. --chain AD:TPS --judge-policy all
```

Livrables dans `out/` :

| Onglet | Contenu |
|---|---|
| `Synthese` | un indicateur par chaîne : taux de traçabilité, propositions, écarts, reuse, rappel/précision mesurés |
| `Matrice` | une ligne par couple (exigence amont, candidat aval) : confiance, lien existant ou non, verdict et justification du juge, détail des 6 signaux |
| `Ecarts` | exigences non couvertes ou partiellement couvertes, avec les **aspects non couverts** listés |
| `Allocation` | exigences allouées à un sous-système sans descendance au niveau inférieur |
| `Liens a auditer` | liens DOORS existants que le moteur ne confirme pas |
| `Fiabilite` | table de calibration : « confiance 0,7–0,8 → x % de vrais liens observés » |
| `Perimetre` / `Parametres` | modules, baselines, dates d'export, modèles, seuils (traçabilité de l'exécution) |

## 4. Reproductibilité

* Tout est dans `config.yaml` : niveaux et motifs de détection, chaînes auditées, mapping
  des colonnes de la vue DOORS, types d'objets considérés comme exigences, poids des
  signaux, seuils (globaux et par chaîne), modèles Ollama, budget LLM.
* Changer de projet = pointer `--data-dir` ailleurs ; changer de vue DOORS = compléter
  `columns` ; ajouter un niveau (ex. `TRS`) = une entrée dans `levels` + une dans `chains`.
* Les embeddings et les réponses du juge sont mis en cache (`.cache/llm.sqlite`,
  clé = modèle + texte) : une seconde exécution est quasi instantanée et donne le même
  résultat (température 0).
* `calibration.json` est versionnable : il fige l'échelle de confiance utilisée.

## 5. Résultats mesurés sur le jeu PKP fourni

10 extractions, 6 877 objets, 3 043 exigences, 13 395 liens existants décodés.
Le moteur est évalué **contre les liens déjà saisis par les ingénieurs** (vérité terrain) :
on ne compte que les candidats réellement trouvés par la recherche.

| Chaîne | Exigences amont tracées (vérité terrain) | Rappel@1 | Rappel@5 | AUC confiance (val. croisée) |
|---|---|---|---|---|
| `CUS → RS` | 203 | 0,80 | 0,95 | 0,76 |
| `RS → AD` | 308 | 0,58 | 0,90 | 0,69 |
| `AD → TPS` | 400 | 0,91 | 0,98 | 0,93 |
| `TPS → EL` | 388 | 0,07 | 0,26 | calibration **refusée** |

Lecture : sur `AD → TPS`, le bon requis aval est le premier proposé 9 fois sur 10 ; sur
`CUS → RS` il est dans les 5 premiers 19 fois sur 20. L'analyste valide une liste courte
au lieu de parcourir ~1 200 objets — c'est là que se gagne le temps d'analyse de couverture.

**Limite assumée et détectée automatiquement** : la chaîne `TPS → EL` (exigence →
fiche de preuve) n'est pas de nature sémantique ; un requis ne « ressemble » pas au
document qui le démontre. Le garde-fou de calibration le détecte (coefficients négatifs)
et **refuse** de produire un taux de confiance plutôt que d'en inventer un. Cette chaîne
doit passer par des règles sur `PlannedDemonstration` / catégorie de document, pas par de
la similarité de texte.

## 6. Choix d'ingénierie notables

* **Pas de réapprentissage des poids.** La vérité terrain ne fournit que des négatifs
  « durs » (candidats proches mais non liés) : un apprentissage multivarié dessus dégrade
  le classement (mesuré : précision@1 0,81 → 0,36). On calibre donc uniquement la
  transformation monotone score → probabilité (Platt, 2 variables : score absolu et
  position relative dans le vivier). Le classement est préservé, la confiance devient
  lisible.
* **Score absolu + position relative.** Le niveau de similarité absolu dépend du style de
  rédaction du référentiel ; la position relative dans le vivier est plus stable. Les deux
  sont nécessaires : le relatif pour classer, l'absolu pour pouvoir déclarer un écart.
* **Aspects non couverts recalculés.** On découpe l'exigence amont en aspects vérifiables
  (puces, phrases modales) et on prend le complémentaire de l'union des aspects couverts :
  on ne fait pas confiance au champ « ce qui manque » renvoyé par le modèle.
* **Appels LLM ciblés et bornés.** `--judge-policy targeted` n'interroge le modèle que là
  où une décision humaine est attendue (propositions, liens douteux, zones grises), pas sur
  les liens déjà confirmés. `--judge-budget` est un plafond **pour tout l'audit**, toutes
  chaînes confondues. Compter ~15 s par exigence jugée avec un 7B et ~60 s avec un 14B :
  un audit complet sans juge prend ~2 min (embeddings en cache), avec 40 appels ~40 min.
  Les réponses étant cachées, relancer l'audit ne les recalcule pas.

## 7. Tests

```bash
python -m pytest tests -q        # 13 tests : parsing, signaux, garde-fous, bout en bout
RTM_DATA=/autre/jeu python -m pytest tests -q
```

Les tests couvrent le décodage des liens DOORS, la normalisation des valeurs chiffrées
(`200 km/h`), le filtrage des hallucinations du juge (candidat ou aspect inexistant,
confiance hors bornes), et un audit complet sans LLM avec seuil de rappel.

## 8. Pistes de suite

1. `TPS → EL` : moteur de règles sur le type de démonstration et la catégorie de document.
2. Lecture directe des `.csv`/`.rtf` DOORS et des modules `TRS` pour couvrir toute la chaîne.
3. Réinjection dans DOORS : export DXL des liens validés (aujourd'hui l'outil ne lit que).
4. Remontée d'un indicateur de qualité de rédaction (l'attribut `AiQualityScore` est déjà lu).
