# TP_Rag_streamlit

> **Agent ReAct (Reasoning + Acting) avec interface Streamlit** — un assistant de
> recherche en IA qui choisit lui-même, à chaque question, entre un **corpus local
> (Chroma)**, une **recherche web (Tavily)** et **arXiv**, puis répond en français
> avec ses sources.

Réécriture en application web du notebook pédagogique
[TP 4.a — Part 1 : Build a ReAct Agent with LangChain](https://github.com/MarouenKsentini/TP-4.a---Part-1-Build-a-ReAct-Agent-with-LangChain) :
mêmes données, même prompt, même golden set — mais un vrai GUI, des traces en direct
et une évaluation du routage en un clic.

![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/streamlit-1.66-FF4B4B?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/langchain-1.4-1C3C3C)
![Tests](https://img.shields.io/badge/tests-53%20passed-brightgreen)
![Licence](https://img.shields.io/badge/licence-pédagogique-lightgrey)

---

## Sommaire

- [Aperçu](#aperçu)
- [Ce que fait l'application](#ce-que-fait-lapplication)
- [Architecture](#architecture)
- [Démarrage rapide](#démarrage-rapide)
- [Configuration des clés](#configuration-des-clés)
- [Guide de l'interface](#guide-de-linterface)
- [Exemples concrets](#exemples-concrets)
- [Golden set et score de routage](#golden-set-et-score-de-routage)
- [Le cœur métier (`agent_core`)](#le-cœur-métier-agent_core)
- [Choix techniques](#choix-techniques)
- [Tests](#tests)
- [Dépannage](#dépannage)
- [Limites et perspectives](#limites-et-perspectives)
- [Structure du dépôt](#structure-du-dépôt)

---

## Aperçu

Le pattern **ReAct** alterne raisonnement et action :

```
Question
   │
   ▼
┌──────────────────────────────────────────────┐
│  Agent ReAct (Groq · gpt-oss-20b)            │
│                                              │
│  Thought ──► tool_calls ?                    │
│                  │ non ──► Réponse + sources │
│                  │ oui                       │
│                  ▼                           │
│  Action : rag_search | tavily_web_search     │
│           | arxiv_search                     │
│                  │                           │
│                  ▼                           │
│  Observation (ToolMessage)                   │
│                  │                           │
│  ◄───────────────┘  (boucle, max 12)         │
└──────────────────────────────────────────────┘
```

Le notebook imprimait cette boucle dans des sorties de cellules. Ici, elle s'affiche
**en direct** dans le chat, puis reste consultable sous forme structurée
(`Action` / `Observation` / `Réponse finale`) avec ses métriques : outils utilisés,
nombre d'étapes, latence.

---

## Ce que fait l'application

| Fonctionnalité | Détail |
| --- | --- |
| 💬 **Chat agentique** | trace `Thought → Action → Observation` streamée pendant l'exécution |
| 🧭 **Routage automatique** | 3 outils, choisis par le LLM via le prompt système |
| 📊 **Golden set** | 5 questions de référence, score de routage automatique, export JSON |
| 🔍 **Trace structurée** | vue lisible + JSON brut d'un run, téléchargeable |
| 📚 **Corpus navigable** | documents sources, chunks, test du retriever avec distances |
| 🩺 **Diagnostic intégré** | paquets, clés, corpus, cache du modèle — dans la sidebar **et** en CLI |
| ⚙️ **Tout réglable** | modèle, température, `max_tokens`, `reasoning_effort`, `recursion_limit`, outils, prompt |
| ♻️ **Robuste** | reprise sur limite TPM (413/429), erreurs visibles, index sans effet de bord |
| 🔐 **Secrets propres** | clés lues dans `.env` / `secrets.toml`, jamais saisies ni affichées dans l'UI |

---

## Architecture

```
tp4a-streamlit/
├── app.py            # UI Streamlit : 4 onglets, caches, session, diagnostic
├── agent_core.py     # métier pur : corpus, index, outils, agent, traces, score, pré-vol
├── check.py          # pré-vol CLI (mêmes contrôles que la sidebar)
├── tests/            # 53 tests hors ligne (métier + UI via AppTest)
├── GUIDE.md          # construction pas à pas (13 étapes)
└── pyproject.toml    # dépendances uv + config pytest
```

Deux règles structurent le code :

1. **`agent_core.py` n'importe jamais Streamlit.** Il est réutilisable tel quel
   (script, notebook, test) et testable sans navigateur.
2. **`app.py` ne contient aucune logique métier.** Il appelle `core.*` et ne fait que
   présenter : widgets, caches, rendu de la trace.

| Couche | Rôle | Dépendances |
| --- | --- | --- |
| `agent_core` | corpus, embeddings, Chroma, outils, LLM, traces, score, pré-vol | langchain 1.x, chromadb, groq, tavily, arxiv |
| `app` | interface, caches, session, diagnostic | streamlit |
| `check` | pré-vol terminal | `agent_core.preflight()` |

---

## Démarrage rapide

Prérequis : [uv](https://docs.astral.sh/uv/), un compte [Groq](https://console.groq.com/keys)
et — facultatif — une clé [Tavily](https://app.tavily.com/home).

```bash
git clone https://github.com/MarouenKsentini/TP_Rag_streamlit.git
cd TP_Rag_streamlit

uv sync                          # Python 3.12 + 164 paquets verrouillés par uv.lock
Copy-Item .env.example .env      # puis coller les clés dans .env
uv run check.py                  # pré-vol : doit finir sur « Prêt »
uv run streamlit run app.py      # http://localhost:8501
```

Sans uv :

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python check.py
streamlit run app.py
```

> Le premier lancement télécharge `all-MiniLM-L6-v2` (~90 Mo) — une seule fois, puis
> le modèle reste en cache local. La page s'affiche immédiatement : le chargement se
> fait à la première question.

---

## Configuration des clés

```dotenv
# .env  (ignoré par git)
GROQ_API_KEY=gsk_votre_cle_groq
TAVILY_API_KEY=tvly-votre_cle_tavily
```

- Priorité : `.streamlit/secrets.toml` → `.env` → variables d'environnement.
- **Aucune clé ne se saisit dans l'interface** : la sidebar affiche seulement le statut
  (`✅ détectée` / `❌ absente`) et la source (`Source : .env`).
- Après avoir modifié `.env`, cliquer sur **♻️ Recharger .env** (pas besoin de redémarrer).
- Sans clé Tavily, l'outil web devient un stub « désactivé » : l'agent et le golden set
  restent utilisables.

> ⚠️ **Piège classique** : coller sa clé dans `.env.example` au lieu de `.env`. Ce
> fichier est versionné et rien ne le lit. Le pré-vol le détecte explicitement :
> `⚠️ clés réelles dans .env.example`. Le bon réflexe reste
> `Copy-Item .env.example .env` **puis** éditer `.env`.

---

## Guide de l'interface

### Barre latérale

| Bloc | Contenu |
| --- | --- |
| 🔑 **Clés API** | statut Groq/Tavily, source du `.env`, bouton *Recharger* |
| 🩺 **Diagnostic** | Configuration, Paquets, Clés API, Corpus, Index — avec la correction à appliquer |
| **Modèle** | nom du modèle Groq, temperature, `max_tokens`, `reasoning_effort`, `recursion_limit` |
| **Outils** | activer/désactiver Tavily (grisé sans clé) et arXiv |
| ⚙️ **Avancé** | prompt système éditable, pause avant reprise TPM, purge des caches |

### Onglets

| Onglet | À quoi ça sert |
| --- | --- |
| 💬 **Agent** | poser une question, voir la boucle se dérouler, lire la réponse + métriques |
| 📊 **Golden set** | lancer les 5 questions, obtenir le score, exporter le run en JSON |
| 🔍 **Trace** | inspecter le dernier run (étapes, JSON brut, téléchargement) |
| 📚 **Corpus** | 4 documents, nombre de chunks, recherche de test avec distances |

---

## Exemples concrets

### 1. Question technique → corpus local, 1 étape

> **« Comment fonctionne l'algorithme HNSW ? »**

```
🧑‍💻 Question
🤖 Action       rag_search({"query": "Comment fonctionne l'algorithme HNSW ?"})
📥 Observation  [source: vector_databases.md] HNSW builds a multi-layer graph…
💬 Réponse      HNSW (Hierarchical Navigable Small World) est un algorithme
                d'indexation et de recherche de voisinage… (sources citées)
```

Métriques mesurées sur cette machine : **1 étape · 2,15 s · reprise TPM : non**.

### 2. Information datée → recherche web

> **« Quels nouveaux modèles d'embedding open-source sont sortis en 2025 ? »**

L'agent appelle `tavily_web_search` (information hors corpus) et cite ses sources.
⚠️ Bon outil ≠ réponse parfaite : la qualification « open-source » peut être
approximative (licences mélangées). C'est exactement la limite pointée par le TP.

### 3. Papier académique → arXiv

> **« Que dit le papier original LoRA (2021) ? »**

`arxiv_search` renvoie date, titre, auteurs et résumé. Sur le run de référence, la
requête est trop vague : l'agent enchaîne avec `rag_search` — auto-correction
**accidentelle**, pas garantie.

### 4. Hors domaine → refus

> **« Quelle est la recette préférée de l'auteur de "Attention is all you need" ? »**

Aucun outil appelé, refus poli en français. C'est le garde-fou du prompt système
(règle simple et non ambiguë = la plus fiable).

### 5. Tester le RAG sans consommer de quota

Onglet **📚 Corpus** → requête `How does HNSW work?` :

```
vector_databases.md · distance 1.4351   ← le bon document arrive en premier
transformers.md     · distance 1.4715
finetuning.md       · distance 1.6568
rag_systems.md      · distance 1.7407
```

---

## Golden set et score de routage

| Q | Question | Outils attendus |
| --- | --- | --- |
| Q1 | Fonctionnement de HNSW | `rag_search` |
| Q2 | Nouveaux embeddings open-source 2025 | `tavily_web_search` |
| Q3 | Papier original LoRA (2021) | `arxiv_search` |
| Q4 | HNSW vs alternatives en production | `rag_search` + `tavily_web_search` |
| Q5 | Recette de l'auteur de *Attention is all you need* | aucun (refus poli) |

**Barème** : `exact` = 1.0 · `partiel` = 0.5 · `faux` = 0.0. Le score est la moyenne
des 5 verdicts. Il mesure **le routage**, pas la qualité du contenu.

Run de référence publié dans le notebook d'origine (`gpt-oss-20b`, `temperature=0`,
`reasoning_effort=low`, `max_tokens=1024`, `recursion_limit=12`) :

| Q | Outils observés | Durée | Étapes | Verdict |
| --- | --- | --- | --- | --- |
| Q1 | `rag_search` | 1,91 s | 1 | ✅ exact |
| Q2 | `tavily_web_search` | 3,61 s | 1 | ✅ exact |
| Q3 | `arxiv_search` + `rag_search` | 21,96 s | 2 | ≈ partiel |
| Q4 | `tavily_web_search` | 16,84 s | 1 | ≈ partiel |
| Q5 | aucun | 11,78 s | 0 | ✅ exact |

**Score : 0,80 / 1,00.**

Non reproductible à l'identique : même à `temperature=0`, le choix des outils et les
latences varient. L'onglet Golden set affiche le run réel, avec export JSON pour le
rapport.

---

## Le cœur métier (`agent_core`)

Le module est utilisable sans Streamlit — utile pour un script ou un notebook :

```python
import agent_core as core
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Index du corpus
store, chunks = core.build_vectorstore(HuggingFaceEmbeddings(model_name=core.EMBEDDING_MODEL))
tools = core.build_tools(retriever=core.build_retriever(store))

# 2. Agent
agent = core.build_agent(core.build_llm(api_key=core.os.environ["GROQ_API_KEY"]), tools)

# 3. Run
result = core.run_react_agent(agent, "Comment fonctionne HNSW ?")
print(result.answer, result.tools_used, result.duration_s)

# 4. Score de routage
verdicts, score = core.routing_score(core.GOLDEN_SET, {"Q1": result})
```

| Fonction | Rôle |
| --- | --- |
| `split_corpus()` | découpe le corpus (tiktoken → caractères → naïf) |
| `build_vectorstore()` / `build_retriever()` | index Chroma en mémoire + retriever `k=4` |
| `make_rag_tool()` / `make_tavily_tool()` / `make_arxiv_tool()` | les trois outils |
| `build_llm()` / `build_agent()` | ChatGroq borné + agent ReAct compilé |
| `run_react_agent()` | exécution, trace, métriques, reprise TPM |
| `steps_from_message()` | messages LangGraph → étapes ReAct |
| `verdict_for()` / `routing_score()` | évaluation du routage |
| `preflight()` | diagnostic complet (partagé avec `check.py`) |

---

## Choix techniques

- **`st.cache_resource` + imports paresseux** — la page s'affiche en moins d'une
  seconde ; `torch`, `chromadb` et `langchain-groq` ne sont importés qu'au premier usage.
- **Arguments de cache = identité de l'agent** — changer modèle, température, outils ou
  prompt recrée l'agent, au lieu de garder un agent obsolète (piège du notebook).
- **Chroma en mémoire** — pas de `persist_directory`, donc pas d'empilement de chunks
  (le notebook passait de 4 → 8 → 12 chunks en ré-exécutant la cellule).
- **Trace lue dans `tool_calls`** — jamais de parsing de texte : c'est la trace
  structurée de LangGraph. `stream(..., stream_mode="values")` alimente l'affichage live.
- **Reprise TPM unique** — tier gratuit Groq = 8 000 TPM ; une erreur `413`/`429`
  déclenche une pause paramétrable puis **une seule** nouvelle tentative.
- **Erreurs exposées** — `RunResult.error` + `st.error`, trace conservée. Le notebook
  renvoyait `"[Erreur : ...]"` avec `tools_used = []`, indistinguable d'une réponse vide.
- **Un seul pré-vol** — `preflight()` sert `check.py` (exit code 1 si bloquant) et la
  sidebar (recalculée à chaque interaction).
- **Trois niveaux de repli pour le découpage** — tiktoken → caractères → fenêtres
  glissantes, pour que l'app reste utilisable hors ligne.

---

## Tests

```bash
uv run pytest -q     # 53 tests, ~30 s, aucun appel réseau
```

| Fichier | Couverture |
| --- | --- |
| `tests/test_agent_core.py` | corpus, golden set, verdicts, score de référence 0,80, messages → trace, `413/429`, reprise unique, streaming, export JSON, pré-vol (`.env` manquant, clés dans `.env.example`, placeholders) |
| `tests/test_app_smoke.py` | rendu Streamlit sans exception, 4 onglets, métriques, diagnostic présent, chat désactivé sans clé, recherche du retriever |

Aucun test n'appelle Groq, Tavily, arXiv, ni ne télécharge de modèle : la suite tourne
hors ligne, en intégration continue ou sur une machine vierge.

---

## Dépannage

| Symptôme | Cause | Solution |
| --- | --- | --- |
| `❌ GROQ_API_KEY absente` alors que la clé est saisie | clé dans `.env.example`, ou `.env` mal placé | `Copy-Item .env.example .env`, coller la clé, **♻️ Recharger .env** |
| `❌ .env introuvable` | fichier absent | même commande ; le pré-vol indique le chemin attendu |
| `404 model_not_found` | modèle indisponible sur le compte | garder `openai/gpt-oss-20b` |
| `413` / `429 rate_limit_exceeded` | plafond gratuit 8 000 TPM | laisser la reprise automatique, baisser `max_tokens` / `reasoning_effort` |
| Premier chargement très long | téléchargement du modèle (~90 Mo) | normal une fois ; ensuite en cache |
| `Impossible de construire l'index vectoriel` | `sentence-transformers` absent / hors ligne | `uv sync` puis relancer |
| `arxiv_search` → « indisponible » | API arXiv momentanément KO | relancer ; l'agent peut basculer d'outil |
| Score < 0,80 | variabilité du routage | relancer et comparer les traces Q3/Q4 |
| Widgets dépréciés | Streamlit trop ancien | `pip install -U streamlit` (≥ 1.40) |

---

## Limites et perspectives

- Un agent ReAct pur **n'a aucun mécanisme d'auto-évaluation** : il ne vérifie ni la
  pertinence des documents, ni la fidélité de sa réponse, et ne change pas de source de
  lui-même en cas d'échec — c'est visible sur Q3 et Q4.
- Le corpus est minuscule (4 documents) et `k=4` renvoie tout le corpus : la pertinence
  du retrieval n'est pas mesurable ici.
- Les questions multi-hop ne sont pas garanties : le prompt reste une **suggestion**,
  pas une contrainte mécanique.
- L'index vit en mémoire : reconstruit à chaque redémarrage du serveur.

La Partie 3 (Adaptive RAG avec LangGraph) corrige ces points avec des nœuds
`grade_docs` / `grade_answer`, une réécriture de requête et un fallback web imposés
par le développeur. Voir aussi [`GUIDE.md`](GUIDE.md) pour la construction détaillée
du projet, étape par étape.

---

## Structure du dépôt

```
TP_Rag_streamlit/
├── app.py                  # interface Streamlit (4 onglets)
├── agent_core.py           # métier : corpus, index, outils, agent, traces, score, pré-vol
├── check.py                # pré-vol CLI
├── GUIDE.md                # construction pas à pas (13 étapes)
├── README.md
├── tests/
│   ├── test_agent_core.py
│   └── test_app_smoke.py
├── .streamlit/config.toml  # thème, télémétrie désactivée
├── .env.example            # placeholders (le vrai .env n'est jamais versionné)
├── .python-version         # 3.12
├── pyproject.toml          # dépendances + pytest
├── uv.lock                 # 164 paquets verrouillés
├── requirements.txt        # équivalent pip
└── .gitignore              # .env, .venv, caches
```

---

Projet pédagogique — TP 4.a, Module 4 « AI Agents ».
Basé sur le notebook de [Marouen Ksentini](https://github.com/MarouenKsentini).
