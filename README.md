# auditeur

Auditeur de traçabilité et de couverture des exigences — PoC 100 % local
(Python + [Ollama](https://ollama.com)), pour environnements industriels où aucune
donnée ne doit sortir du poste.

Il compare plusieurs extractions de référentiels **IBM DOORS** (`.CUS`, `.RS`, `.AD`,
`.RSTPS`, `.EL`), produit une **matrice des liens possibles** entre niveaux
(`.CUS → RSL0 → RSL1 → TPS`), **propose un ou plusieurs requis à chaîner avec un taux de
confiance calibré**, et **liste les écarts** quand une partie de l'exigence n'est pas
couverte.

➡️ Documentation, installation, usage et résultats mesurés : **[rtm-auditor/README.md](rtm-auditor/README.md)**

```bash
cd rtm-auditor
python -m rtm.cli inspect  --data-dir /chemin/vers/extractions
python -m rtm.cli calibrate --data-dir /chemin/vers/extractions --all-chains
python -m rtm.cli audit     --data-dir /chemin/vers/extractions --all-chains
```

## Confidentialité

Ce dépôt ne contient **que l'outil**. Les extractions DOORS, les rapports produits et le
cache des modèles restent locaux : ils sont exclus par [.gitignore](.gitignore) et ne
doivent jamais être publiés. L'outil n'émet aucune requête réseau hors de
`http://127.0.0.1:11434` (Ollama).
