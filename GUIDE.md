# Construction pas à pas — du notebook à l'application Streamlit

Ce guide décrit **comment le projet a été fabriqué**, étape par étape, avec le
raisonnement derrière chaque choix. Le résultat est dans `app.py` + `agent_core.py`.

Pour simplement **installer et lancer**, voir le §3 du [README](README.md).

---

## Étape 1 — Séparer le métier de l'interface

**Objectif** : rendre la logique réutilisable (CLI, notebook, tests, UI).

```
agent_core.py   ← corpus, outils, agent, traces, score, pré-vol   (0 import Streamlit)
app.py          ← widgets, mise en page, caches                    (0 logique métier)
```

Règle appliquée : `app.py` ne contient **aucun** appel LLM direct, seulement des appels
à `core.*`. C'est ce qui permet de tester le score et les traces sans navigateur.

```python
# app.py
import agent_core as core

result = core.run_react_agent(agent, question, recursion_limit=12)
```

---

## Étape 2 — Reprendre le corpus, le prompt et le golden set à l'identique

Le notebook était la référence pédagogique : on ne change **pas** les données, sinon
les résultats ne sont plus comparables.

| Élément | Emplacement | Modifié ? |
| --- | --- | --- |
| 4 documents IA | `CORPUS` | non |
| prompt de routage | `SYSTEM_PROMPT` | non (éditable dans l'UI) |
| 5 questions + outils attendus | `GOLDEN_SET` | non |
| sémantique du score | `verdict_for` / `routing_score` | non |

---

## Étape 3 — Indexer le corpus sans effet de bord

Le notebook devait supprimer la collection Chroma pour éviter l'empilement de chunks
(4 → 8 → 12). En application, on supprime la cause : **pas de `persist_directory`**,
donc un index neuf en mémoire à chaque démarrage.

```python
def build_vectorstore(embeddings, *, collection_name=COLLECTION_NAME):
    chunks = split_corpus()                                  # tiktoken → chars → naïf
    store = Chroma.from_documents(chunks, embeddings, collection_name=collection_name)
    return store, chunks
```

Trois niveaux de repli pour le découpage : `RecursiveCharacterTextSplitter` + tiktoken,
sinon en caractères, sinon fenêtres glissantes. L'app reste utilisable hors ligne.

---

## Étape 4 — Déclarer les trois outils

```python
@tool                                   # ← le nom et la docstring sont lus par le LLM
def rag_search(query: str) -> str: ...
```

- `@tool` plutôt que `Tool(func=...)` : la trace affiche `{'query': ...}` et non `{'__arg1': ...}`.
- **Tavily** : `TavilySearch` (LangChain 1.x) renvoie `{"results": [...]}`. Sans clé, on
  renvoie un `Tool` stub qui répond « désactivé » pour garder 3 outils et un agent valide.
- **arXiv** : SDK `arxiv` + `@tool` (remplace `ArxivQueryRun`, déprécié). Les erreurs
  réseau deviennent une observation texte, jamais une exception.

---

## Étape 5 — Construire l'agent avec des réglages bornés

```python
llm = ChatGroq(model=model, temperature=0, max_tokens=1024, reasoning_effort="low")
agent = create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)   # LangChain 1.x → LangGraph
```

`reasoning_effort` + `max_tokens` ne sont pas cosmétiques : le tier gratuit plafonne à
8 000 TPM, et sans bornes le golden set renvoie `413 rate_limit_exceeded`.

`reasoning_effort="désactivé"` dans l'UI ⇒ le paramètre n'est pas transmis (utile pour
les modèles Groq non « reasoning »).

---

## Étape 6 — Transformer les messages en trace lisible

Un run LangGraph renvoie `state["messages"]` :
`HumanMessage → AIMessage(tool_calls) → ToolMessage → … → AIMessage final`.

```python
steps_from_message(msg)  →  [TraceStep(kind="action", name="rag_search", args={...})]
```

Trois conséquences :

- la trace est **structureé** (lue dans `tool_calls`, jamais parsée depuis du texte) ;
- `tools_used` et `steps` en découlent, donc les métriques ne peuvent pas diverger de la trace ;
- `agent.stream(..., stream_mode="values")` donne les mêmes étapes **en direct** dans le chat.

Un `AIMessage` qui contient à la fois du texte et des `tool_calls` produit un
`thought` puis des `action` : c'est le rendu ReAct classique.

---

## Étape 7 — Mesurer le routage

```python
verdict_for(expected, observed)   # exact / partiel / faux
routing_score(golden_set, results)  # 1.0 / 0.5 / 0.0 par question, moyenne arrondie
```

Règles conservées du notebook : ensembles égaux = `exact` ; intersection non vide =
`partiel` ; sinon `faux`. Cas particulier : refus attendu (`[]`) + outil appelé = `faux`.
Le scénario de référence (Q1,Q2,Q5 exacts ; Q3,Q4 partiels) donne **0,80** — c'est
verrouillé par un test.

---

## Étape 8 — Brancher Streamlit

Trois mécanismes suffisent :

| Besoin | Mécanisme |
| --- | --- |
| modèle d'embedding chargé une fois | `@st.cache_resource` (clé = objet ignoré `_embeddings`) |
| agent reconstruit si un réglage change | `@st.cache_resource` dont les **arguments sont la clé** (modèle, temperature, outils, prompt, clés) |
| conversation qui survit aux reruns | `st.session_state["history"]` (liste de `RunResult`) |

Les imports lourds (`sentence-transformers`, `chromadb`, `langchain-groq`) sont **à
l'intérieur** des fonctions : la page s'affiche en < 1 s, le modèle (~90 Mo) n'est
chargé qu'à la première question.

Le chat utilise un `st.form` (question + bouton) : l'envoi déclenche un seul run, puis
`st.rerun()` réaffiche l'historique complet.

---

## Étape 9 — Rendre les échecs visibles

Le notebook avalait toute exception dans `"[Erreur : ...]"` avec `tools_used = []` :
un échec ressemblait à une réponse vide.

```python
@dataclass
class RunResult:
    answer: str; tools_used: list[str]; steps: int; duration_s: float
    trace: list[TraceStep]; error: str | None; retried: bool
```

- l'erreur est stockée **et** affichée (`st.error`) ;
- la trace reste consultable même en échec ;
- une limite TPM déclenche `on_retry` → avertissement + pause + **une seule** reprise.

---

## Étape 10 — Un pré-vol unique, deux affichages

`agent_core.preflight()` renvoie une liste de `CheckItem` (groupe, libellé, verdict,
détail, correction, criticité). Deux consommateurs :

```bash
uv run check.py            # CLI, code de sortie 1 si un point bloquant subsiste
```

```
🩺 Diagnostic                # barre latérale Streamlit, recalculé à chaque interaction
```

Aucun cache volontairement : créer `.env` doit changer le verdict immédiatement, sans
redémarrer l'application (bouton **♻️ Recharger .env** → `reload_env()` + purge du cache
des ressources).

Le groupe **Configuration** est celui qui attrape l'erreur la plus fréquente du TP :
une clé collée dans `.env.example` au lieu de `.env`.

```python
check_env_file()        # .env existe ? chargé depuis où ? clés réelles dans l'exemple ?
```

Une clé n'est jamais saisie dans l'interface : la sidebar n'expose que le **statut** et
la source (`Source : .env`). Le code lit `.streamlit/secrets.toml` puis l'environnement
(rempli par `dotenv`), donc une clé ne peut pas se retrouver dans un fichier versionné
par accident d'UI.

---

## Étape 11 — Verrouiller l'environnement avec uv

```
pyproject.toml   → dépendances déclarées (streamlit, langchain 1.x, chroma, torch…)
.python-version  → 3.12
uv.lock          → 164 paquets résolus, versions figées
```

```bash
uv sync            # crée .venv/ exactement comme uv.lock
uv sync --upgrade  # met à jour et réécrit uv.lock
```

`[tool.uv] package = false` : le projet est une **application**, pas une bibliothèque
publiée — uv n'essaie donc pas de la construire.

---

## Étape 12 — Tester sans dépenser de quota

```bash
uv run pytest -q     # 53 tests, ~25 s, aucun appel réseau
```

| Famille | Exemple |
| --- | --- |
| métier | `routing_score` sur le scénario de référence = 0,80 |
| traces | `AIMessage(tool_calls=[...])` → `thought` + `action` |
| robustesse | agent qui lève `413` → une seule reprise, `retried=True` |
| streaming | faux agent `.stream()` → étapes reçues dans l'ordre |
| pré-vol | clé `VOTRE_CLE_GROQ_ICI` rejetée, Tavily non bloquant |
| UI | `AppTest` : 4 onglets, diagnostic présent, aucun crash |

`AppTest` (livré avec Streamlit) exécute `app.py` sans navigateur : c'est le test qui
attrape les erreurs de widgets après une mise à jour de Streamlit.

---

## Étape 13 — Vérifier de bout en bout

```bash
uv run check.py                      # 16/18 : il ne manque que GROQ_API_KEY
uv run streamlit run app.py          # http://localhost:8501
```

Dans l'app : onglet **Corpus** → « Construire / recharger l'index » puis une requête
`How does HNSW work?` doit renvoyer `vector_databases.md` en premier (distance la plus
faible). C'est le test du RAG local **sans** consommer de quota LLM.

Avec la clé Groq : onglet **Agent** (une question) puis onglet **Golden set**
(score + export JSON).

---

## Récapitulatif des fichiers

| Fichier | Rôle | Étape |
| --- | --- | --- |
| `agent_core.py` | corpus, index, outils, agent, trace, score, pré-vol | 1–10 |
| `app.py` | 4 onglets, caches, session, diagnostic | 8–10 |
| `check.py` | pré-vol CLI | 10 |
| `tests/` | métier + UI sans réseau | 12 |
| `pyproject.toml`, `uv.lock`, `.python-version` | environnement reproductible | 11 |
| `.env.example`, `.streamlit/config.toml` | configuration | 3 |
