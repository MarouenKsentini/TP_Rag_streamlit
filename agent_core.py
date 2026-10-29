"""Cœur du ReAct agent — aucune dépendance à Streamlit.

Ce module est réutilisable tel quel : l'application Streamlit (`app.py`), un
script CLI ou un notebook consomment les mêmes fonctions. Les imports lourds
(sentence-transformers, chromadb, groq, tavily, arxiv) sont faits *à l'intérieur*
des fonctions pour garder l'import du module rapide et testable hors ligne.

Sommaire du module
------------------
    1. Chargement de .env et source des clés
    2. Constantes (modèle, embedding, chunking)
    3. Corpus local + prompt système + golden set
    4. Corpus → Documents → chunks → index Chroma → retriever
    5. Les trois outils : rag_search, tavily_web_search, arxiv_search
    6. LLM + agent ReAct (LangChain 1.x)
    7. Messages LangGraph → trace lisible (`TraceStep`, `RunResult`)
    8. Exécution (`run_react_agent`) avec reprise sur limite de quota
    9. Évaluation du routage (`verdict_for`, `routing_score`)
   10. Pré-vol / diagnostic (`preflight`, partagé avec `check.py`)
"""

from __future__ import annotations

import importlib.util
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

# --- Variables d'environnement Hugging Face ---------------------------------
# À définir AVANT tout import de transformers / sentence-transformers, sinon les
# avertissements « unauthenticated requests to the HF Hub » polluent la console.
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import find_dotenv, load_dotenv
from langchain_core.documents import Document
from langchain_core.tools import Tool, tool

ENV_FILE = ".env"
ENV_EXAMPLE_FILE = ".env.example"

ENV_PATH = ""  # chemin absolu du .env réellement chargé (vide si absent)


def _load_env(path: str | None = None) -> str:
    """Charge .env et mémorise le fichier utilisé. Ne casse jamais le démarrage.

    `override=False` quand on lit au démarrage : une variable déjà exportée dans le
    shell reste prioritaire. `override=True` uniquement sur rechargement explicite
    (bouton « Recharger .env »), sinon la nouvelle valeur serait ignorée.
    """
    global ENV_PATH
    try:
        loaded = load_dotenv(dotenv_path=path, override=path is not None)
        found = find_dotenv(usecwd=True) if loaded else ""
        ENV_PATH = os.path.abspath(found) if found else ""
    except Exception:  # fichier illisible, encodage exotique… on continue sans .env
        ENV_PATH = ""
    return ENV_PATH


_load_env()  # au démarrage : ./.env n'écrase pas les variables déjà exportées


def reload_env(path: str | None = None) -> str:
    """Recharge .env (bouton « Recharger .env » / après création du fichier)."""
    return _load_env(path)


def env_source() -> str:
    """Libellé lisible de la source des clés, sans jamais afficher leur valeur.

    On n'expose que le **chemin** du fichier chargé : assez pour diagnostiquer
    (« ah, il lit .env et pas mon fichier ailleurs »), jamais le secret.
    """
    if not ENV_PATH:
        return "variables d'environnement (aucun .env trouvé)"
    try:
        return os.path.relpath(ENV_PATH)
    except ValueError:  # disques différents sous Windows (C: vs D:)
        return ENV_PATH


# --- Constantes --------------------------------------------------------------
# Modèle du sujet : llama-3.3-70b-versatile → 404 sur ce compte (offre entreprise).
# gpt-oss-20b est le modèle disponible ; garder les tirets, sinon 404 model_not_found.
DEFAULT_MODEL = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # petit (~90 Mo), rapide, suffisant pour 4 docs
COLLECTION_NAME = "tp4a_corpus"  # nom de collection Chroma (index en mémoire)
CHUNK_SIZE = 300  # tokens : compromis précision / contexte (cf. corpus RAG)
CHUNK_OVERLAP = 30  # 10 % de recouvrement pour ne pas couper une idée en deux
DEFAULT_K = 4  # 4 chunks pour 4 documents = tout le corpus (voulu par le sujet)
DEFAULT_RECURSION_LIMIT = 12  # garde-fou anti-boucle de l'agent

# --- Corpus local (identique au notebook) ------------------------------------
CORPUS: dict[str, str] = {
    "transformers.md": """# Transformer Architecture
## Self-Attention
The Transformer architecture (Vaswani et al., 2017) replaced RNNs with self-attention.
It computes Query (Q), Key (K), Value (V) matrices and uses softmax(QK^T / sqrt(d_k)) * V.
Multi-head attention runs multiple attention heads in parallel.
## Variants
Encoder-only models (BERT), decoder-only models (GPT), encoder-decoder (T5, BART).
Modern positional encodings: RoPE (Rotary), ALiBi.""",
    "rag_systems.md": """# Retrieval-Augmented Generation (RAG)
## Definition
RAG combines an information retrieval layer with an LLM. Responses are generated from
retrieved evidence rather than parametric memory only.
## Hybrid Retrieval
Hybrid retrieval combines sparse (BM25) and dense (embeddings) methods, fused with
Reciprocal Rank Fusion (RRF).
## Chunking
Common strategies: fixed-size, recursive, semantic, structure-aware (Markdown headers).
Optimal chunk_size depends on the use case: 256 tokens for precision, 1024 for context.""",
    "vector_databases.md": """# Vector Databases and Indexing
## HNSW (Hierarchical Navigable Small World)
HNSW builds a multi-layer graph where each layer is a small-world network.
Search starts at the top sparse layer and refines through lower dense layers.
Achieves 95%+ recall with millisecond latency. Key params: M, ef_construction, ef_search.
## IVF (Inverted File Index)
IVF clusters vectors with k-means; search only in nearest clusters (nprobe).
Combined with PQ (product quantization) for memory savings on huge corpora.""",
    "finetuning.md": """# LLM Fine-Tuning
## LoRA (Low-Rank Adaptation)
LoRA freezes the pre-trained model and injects trainable rank decomposition matrices
into each Transformer layer. Update is W + BA where B,A are small.
## QLoRA
QLoRA combines LoRA with 4-bit quantization. Enables fine-tuning 65B models on a single 48GB GPU.
## RLHF and DPO
RLHF: reward model + PPO. DPO: simpler, no separate reward model, classification loss on preference pairs.""",
}

SYSTEM_PROMPT = """Tu es un AI Research Assistant spécialisé en intelligence artificielle.
Tu disposes de 3 outils :
- `rag_search` : corpus IA local (Transformer, RAG, HNSW, LoRA, vector DBs).
- `tavily_web_search` : recherche web généraliste (infos récentes 2024-2026).
- `arxiv_search` : papiers scientifiques arXiv (publications académiques précises).

RÈGLES :
- Utilise `rag_search` en priorité pour les concepts fondamentaux IA déjà bien établis
  (Transformer, RAG, HNSW, LoRA, embeddings...).
- Utilise `tavily_web_search` pour les informations récentes (2024-2026) ou hors corpus local.
- Utilise `arxiv_search` quand la question concerne un papier précis ou des publications
  académiques.
- Refuse poliment les questions hors-domaine (cuisine, météo, etc.) en français,
  SANS appeler aucun outil.
- Cite tes sources dans ta réponse finale.
- Réponds en français.
"""

# Golden set = jeu d'évaluation stable : questions + outils attendus.
GOLDEN_SET: list[dict[str, Any]] = [
    {
        "id": "Q1",
        "question": "Comment fonctionne l'algorithme HNSW pour la recherche approximative de plus proches voisins ?",
        "expected_tools": ["rag_search"],
    },
    {
        "id": "Q2",
        "question": "Quels nouveaux modèles d'embedding open-source sont sortis en 2025 ?",
        "expected_tools": ["tavily_web_search"],
    },
    {
        "id": "Q3",
        "question": "Que dit le papier original LoRA (2021) sur l'efficacité paramétrique du fine-tuning ?",
        "expected_tools": ["arxiv_search"],
    },
    {
        "id": "Q4",
        "question": "Compare HNSW à des alternatives utilisées en production en 2024-2025.",
        "expected_tools": ["rag_search", "tavily_web_search"],
    },
    {
        "id": "Q5",
        "question": "Quelle est la recette préférée de l'auteur du papier 'Attention is all you need' ?",
        "expected_tools": [],
    },
]


# --- Corpus / index ----------------------------------------------------------
def build_documents() -> list[Document]:
    """Transforme le corpus en `Document` en conservant la source en métadonnée."""
    return [Document(page_content=text, metadata={"source": name}) for name, text in CORPUS.items()]


def split_corpus(
    *,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    use_tiktoken: bool = True,
) -> list[Document]:
    """Découpe le corpus en chunks de ~300 tokens.

    Trois niveaux de repli, du plus fidèle au plus robuste :
    1. `RecursiveCharacterTextSplitter` avec l'encodeur tiktoken (notebook) ;
    2. le même splitter en caractères, si tiktoken est indisponible (hors ligne) ;
    3. un découpage naïf en fenêtres glissantes, si `langchain-text-splitters`
       n'est pas installé (l'app reste utilisable, l'index est juste plus grossier).
    """
    docs = build_documents()
    if use_tiktoken:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            # Découpage « du notebook » : la taille est comptée en tokens cl100k_base.
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            return splitter.split_documents(docs)
        except Exception:  # tiktoken indisponible (hors ligne) → repli suivant
            pass

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # Repli : mêmes réglages exprimés en caractères (~4 caractères / token).
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size * 4, chunk_overlap=chunk_overlap * 4
        )
        return splitter.split_documents(docs)
    except Exception:
        return _naive_split(docs, chunk_size * 4, chunk_overlap * 4)


def _naive_split(docs: list[Document], size: int, overlap: int) -> list[Document]:
    """Dernier recours : fenêtres glissantes de `size` caractères."""
    chunks: list[Document] = []
    step = max(1, size - overlap)
    for doc in docs:
        text = doc.page_content
        for start in range(0, max(len(text), 1), step):
            piece = text[start : start + size].strip()
            if piece:
                chunks.append(Document(page_content=piece, metadata=dict(doc.metadata)))
    return chunks


def build_vectorstore(
    embeddings: Any,
    *,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    collection_name: str = COLLECTION_NAME,
) -> tuple[Any, list[Document]]:
    """Construit un index Chroma *en mémoire* et renvoie (store, chunks).

    Pas de `persist_directory` : l'index est reconstruit à chaque démarrage à
    froid, ce qui évite l'empilement de chunks (4→8→12) constaté dans le notebook.
    """
    from langchain_chroma import Chroma

    chunks = split_corpus(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # `collection_name` explicite : sans lui, Chroma en génère un aléatoire et il
    # devient impossible de retrouver/écraser la collection en cas de besoin.
    store = Chroma.from_documents(chunks, embeddings, collection_name=collection_name)
    return store, chunks


def build_retriever(store: Any, *, k: int = DEFAULT_K) -> Any:
    """k=4 avec 4 documents = tout le corpus renvoyé, seul l'ordre varie (voulu)."""
    return store.as_retriever(search_kwargs={"k": k})


# --- Outils ------------------------------------------------------------------
def make_rag_tool(retriever: Any) -> Any:
    """Outil n°1 : le corpus local (Chroma)."""

    # `@tool` plutôt que `Tool(func=...)` : le nom de l'outil est celui de la
    # fonction et l'argument reste `query`. Avec un Tool générique, la trace
    # afficherait `rag_search({'__arg1': ...})`.
    @tool
    def rag_search(query: str) -> str:
        """Cherche des informations dans le corpus IA local (Transformer, RAG,
        fine-tuning, vector databases). À utiliser pour les questions techniques
        sur ces sujets quand l'information est susceptible d'être dans la base."""
        # ATTENTION : cette docstring EST la description lue par le LLM pour choisir
        # l'outil. Elle décrit *quand* l'utiliser, pas seulement ce qu'il fait.
        docs = retriever.invoke(query)
        if not docs:
            return "[Aucun document pertinent trouvé dans le corpus local]"
        # Chaque extrait est préfixé par sa source : c'est ce qui permet au LLM de
        # citer (et à l'utilisateur de vérifier) au lieu d'inventer.
        return "\n\n---\n\n".join(
            f"[source: {d.metadata.get('source', '?')}]\n{d.page_content}" for d in docs
        )

    return rag_search


def make_tavily_tool(api_key: str = "", *, max_results: int = 3) -> Any:
    """Outil n°2 : recherche web. Renvoie un stub si la clé est absente.

    Pourquoi un stub et pas `None` ? Pour que la liste d'outils garde toujours 3
    entrées : l'agent démarre, le prompt reste cohérent, et le golden set est
    exécutable sans clé Tavily (l'observation dira simplement « désactivé »).
    """
    if not api_key:
        return Tool.from_function(
            func=lambda q: "[Tavily désactivé — pas de clé API]",
            name="tavily_web_search",
            description="Recherche web (désactivée).",
        )

    from langchain_tavily import TavilySearch

    # TavilySearch lit la clé dans l'environnement (client interne) : on la republie
    # ici pour que le comportement ne dépende pas de l'ordre d'initialisation.
    os.environ["TAVILY_API_KEY"] = api_key
    # LangChain 1.x : TavilySearch renvoie {"results": [...]}, pas une liste
    # (piège classique quand on affiche brut l'observation dans une trace).
    return TavilySearch(
        max_results=max_results,
        name="tavily_web_search",
        description=(
            "Recherche sur le web public. À utiliser pour les questions récentes, "
            "hors du corpus local, ou nécessitant des informations à jour."
        ),
    )


def make_arxiv_tool(*, max_results: int = 3) -> Any:
    """Outil n°3 : arXiv via le SDK officiel (remplace ArxivQueryRun, déprécié)."""
    import arxiv

    @tool
    def arxiv_search(query: str) -> str:
        """Cherche des papiers scientifiques sur arXiv. À utiliser pour les questions
        concernant des publications académiques précises (auteur, titre, papier précis)."""
        try:
            papers = list(arxiv.Client().results(arxiv.Search(query=query, max_results=max_results)))
        except Exception as exc:
            # Réseau / API indisponible : on renvoie un texte, on ne lève pas.
            # La boucle ReAct continue et l'agent peut basculer sur un autre outil.
            return f"[arXiv indisponible : {exc}]"
        if not papers:
            return "[Aucun papier arXiv trouvé pour cette requête]"
        # Date + titre + auteurs + résumé : de quoi citer un papier précisément.
        return "\n\n---\n\n".join(
            f"[{p.published:%Y-%m-%d}] {p.title}\n"
            f"Auteurs : {', '.join(a.name for a in p.authors)}\n{p.summary.strip()}"
            for p in papers
        )

    return arxiv_search


def build_tools(
    *,
    retriever: Any,
    tavily_key: str = "",
    enable_web: bool = True,
    enable_arxiv: bool = True,
) -> list[Any]:
    """Liste d'outils exposée au LLM (l'ordre n'est pas significatif)."""
    tools: list[Any] = [make_rag_tool(retriever)]
    if enable_web:
        tools.append(make_tavily_tool(tavily_key))
    if enable_arxiv:
        tools.append(make_arxiv_tool())
    return tools


# --- LLM / agent -------------------------------------------------------------
def build_llm(
    *,
    model: str = DEFAULT_MODEL,
    api_key: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    reasoning_effort: str | None = "low",
) -> Any:
    """ChatGroq borné : le tier gratuit plafonne à 8 000 TPM (413 sinon).

    `reasoning_effort=None` ⇒ paramètre non transmis (modèles Groq classiques qui
    rejetteraient un argument inconnu).
    """
    from langchain_groq import ChatGroq

    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "temperature": temperature,  # 0 = reproductibilité maximale (pas absolue)
        "max_tokens": max_tokens,  # borne la consommation de TPM
    }
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    return ChatGroq(**kwargs)


def build_agent(llm: Any, tools: Iterable[Any], system_prompt: str = SYSTEM_PROMPT) -> Any:
    """Compile l'agent ReAct (LangChain 1.x renvoie un graphe LangGraph).

    `system_prompt=` remplace l'ancien `prompt=` ; le retour n'est plus un
    AgentExecutor mais un graphe compilé (invoke/stream sur `{"messages": [...]}`).
    Modifier le prompt ou la liste d'outils impose de reconstruire l'agent.
    """
    from langchain.agents import create_agent

    return create_agent(llm, list(tools), system_prompt=system_prompt)


# --- Traces / exécution ------------------------------------------------------
def text_of(content: Any) -> str:
    """Normalise le contenu d'un message (str, liste de blocs, None)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


@dataclass
class TraceStep:
    """Une ligne de la boucle ReAct lisible par l'UI."""

    kind: str  # human | thought | action | observation | final
    content: str = ""
    name: str | None = None
    args: dict[str, Any] = field(default_factory=dict)

    @property
    def icon(self) -> str:
        return {
            "human": "🧑‍💻",
            "thought": "🧠",
            "action": "🤖",
            "observation": "📥",
            "final": "💬",
        }.get(self.kind, "•")

    @property
    def label(self) -> str:
        return {
            "human": "Question",
            "thought": "Thought",
            "action": "Action",
            "observation": "Observation",
            "final": "Réponse finale",
        }.get(self.kind, self.kind)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"kind": self.kind, "content": self.content}
        if self.name:
            data["name"] = self.name
        if self.args:
            data["args"] = self.args
        return data


@dataclass
class RunResult:
    """Résultat d'un run : réponse, métriques et trace complète."""

    answer: str = ""
    tools_used: list[str] = field(default_factory=list)
    steps: int = 0
    duration_s: float = 0.0
    trace: list[TraceStep] = field(default_factory=list)
    error: str | None = None
    retried: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "tools_used": self.tools_used,
            "steps": self.steps,
            "duration_s": self.duration_s,
            "retried": self.retried,
            "error": self.error,
            "trace": [step.to_dict() for step in self.trace],
        }


def steps_from_message(message: Any) -> list[TraceStep]:
    """Convertit un message LangGraph en étapes ReAct affichables.

    On lit les `tool_calls` structurés (JSON) et jamais du texte « Thought: » /
    « Action: » : c'est la trace officielle du graphe, donc fiable.

    Un AIAssistant peut porter du texte ET des appels d'outils : on produit alors
    un `thought` (le raisonnement) suivi des `action` (les appels).
    """
    cls = type(message).__name__
    if cls == "HumanMessage":
        return [TraceStep(kind="human", content=text_of(getattr(message, "content", "")))]

    if cls == "AIMessage":
        content = text_of(getattr(message, "content", ""))
        calls = list(getattr(message, "tool_calls", None) or [])
        steps: list[TraceStep] = []
        if content and calls:
            steps.append(TraceStep(kind="thought", content=content))
        for call in calls:
            steps.append(
                TraceStep(kind="action", name=call.get("name"), args=call.get("args") or {})
            )
        # Un AIMessage sans outil et avec du texte = la réponse finale.
        # Un AIMessage vide (fréquent avec les modèles « reasoning ») est ignoré.
        if not calls and content:
            steps.append(TraceStep(kind="final", content=content))
        return steps

    if cls == "ToolMessage":
        return [
            TraceStep(
                kind="observation",
                name=getattr(message, "name", None) or "tool",
                content=text_of(getattr(message, "content", "")),
            )
        ]

    return []


def is_rate_limited(result: RunResult) -> bool:
    """Détecte une limite de quota Groq (413/429) pour déclencher une reprise.

    On cherche dans l'erreur ET dans la réponse : selon l'endroit où l'exception
    survient, le message peut se retrouver dans l'un ou dans l'autre.
    """
    haystack = f"{result.error or ''} {result.answer}"
    lowered = haystack.lower()
    return any(
        marker in lowered
        for marker in ("rate_limit", "rate limit", "413", "429", "tokens per minute")
    )


def run_react_agent(
    agent: Any,
    question: str,
    *,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
    on_step: Callable[[TraceStep], None] | None = None,
    retry_on_rate_limit: bool = True,
    retry_delay: float = 60.0,
    on_retry: Callable[[float], None] | None = None,
) -> RunResult:
    """Exécute la boucle Thought → Action → Observation → Réponse.

    `on_step` reçoit chaque étape dès qu'elle est produite (mode streaming), ce qui
    permet à l'UI d'afficher la boucle en direct. `on_retry` est appelé avant la
    pause de 60 s : l'UI peut alors afficher un avertissement bloquant.
    """
    result = _invoke_once(agent, question, recursion_limit=recursion_limit, on_step=on_step)
    if retry_on_rate_limit and is_rate_limited(result):
        # Une seule reprise : au-delà, on consommerait du quota pour rien et on
        # masquerait un problème réel (clé invalide, modèle indisponible…).
        if on_retry is not None:
            on_retry(retry_delay)  # l'UI affiche l'avertissement et gère l'attente
        else:
            time.sleep(retry_delay)
        result = _invoke_once(agent, question, recursion_limit=recursion_limit, on_step=on_step)
        result.retried = True
    return result


def _invoke_once(
    agent: Any,
    question: str,
    *,
    recursion_limit: int,
    on_step: Callable[[TraceStep], None] | None = None,
) -> RunResult:
    """Un seul passage dans le graphe, avec ou sans streaming.

    - `on_step is None` → `invoke()` : on récupère l'état final d'un coup.
    - sinon → `stream(..., stream_mode="values")` : chaque étape du graphe renvoie
      l'état complet et cumulé ; on n'émet que les messages nouveaux (différence
      avec le compteur `seen`).
    """
    t0 = time.perf_counter()
    trace: list[TraceStep] = []
    messages: list[Any] = []

    def record(message: Any) -> None:
        # Un même message peut produire plusieurs étapes (ex. 2 tool_calls).
        for step in steps_from_message(message):
            trace.append(step)
            if on_step is not None:
                on_step(step)

    try:
        payload = {"messages": [("user", question)]}
        config = {"recursion_limit": recursion_limit}  # anti-boucle
        if on_step is None:
            state = agent.invoke(payload, config)
            messages = list(state.get("messages", []))
            for message in messages:
                record(message)
        else:
            seen = 0
            last_state: Mapping[str, Any] | None = None
            for state in agent.stream(payload, config, stream_mode="values"):
                last_state = state
                current = list(state.get("messages", []))
                if len(current) < seen:  # sécurité si le flux n'est pas cumulatif
                    seen = 0
                for message in current[seen:]:
                    seen += 1
                    record(message)
                messages = current
            if last_state is None:  # flux vide (ne devrait pas arriver)
                messages = []
    except Exception as exc:
        # On n'avale jamais l'erreur : elle est stockée ET renvoyée à l'appelant.
        # Le notebook d'origine la transformait en réponse vide, ce qui masquait
        # une clé invalide ou un modèle inexistant.
        return RunResult(
            answer=f"[Erreur : {exc}]",
            tools_used=[],
            steps=0,
            duration_s=round(time.perf_counter() - t0, 2),
            trace=trace,
            error=str(exc),
        )

    duration = round(time.perf_counter() - t0, 2)
    # 1 étape = 1 appel d'outil : c'est la métrique affichée par l'UI.
    tools_used = [step.name for step in trace if step.kind == "action" and step.name]

    # Réponse finale = dernier `final` non vide (le plus récent gagne).
    answer = ""
    for step in reversed(trace):
        if step.kind == "final" and step.content:
            answer = step.content
            break

    return RunResult(
        answer=answer,
        tools_used=tools_used,
        steps=len(tools_used),
        duration_s=duration,
        trace=trace,
        error=None,
    )


# --- Évaluation --------------------------------------------------------------
def verdict_for(expected: Iterable[str], observed: Iterable[str]) -> str:
    """exact : mêmes outils · partiel : intersection non vide · faux : sinon.

    Les ensembles ignorent l'ordre et les doublons : appeler deux fois
    `rag_search` ne change pas le verdict.
    Cas particulier : refus attendu (`[]`) mais outil appelé → `faux`.
    """
    exp, got = set(expected), set(observed)
    if exp == got:
        return "exact"
    if exp and got and (exp & got):
        return "partiel"
    return "faux"


def _tools_of(result: Any) -> list[str]:
    """Accepte un RunResult, un dict sérialisé ou n'importe quel objet `tools_used`."""
    if isinstance(result, RunResult):
        return list(result.tools_used)
    if isinstance(result, Mapping):
        return list(result.get("tools_used") or [])
    return list(getattr(result, "tools_used", []) or [])


def routing_score(
    golden_set: Iterable[Mapping[str, Any]],
    results: Mapping[str, Any],
) -> tuple[dict[str, str], float]:
    """Compare outils attendus et observés ; renvoie (verdicts par question, score).

    Score = moyenne des points : exact 1.0, partiel 0.5, faux 0.0.
    Il mesure **le routage uniquement** : une réponse peut être « exacte » en
    outils et fausse en contenu (cf. Q2/Q3 du rapport).
    """
    golden = list(golden_set)
    points = {"exact": 1.0, "partiel": 0.5, "faux": 0.0}
    per_q: dict[str, str] = {}
    total = 0.0
    for item in golden:
        verdict = verdict_for(item["expected_tools"], _tools_of(results[item["id"]]))
        per_q[item["id"]] = verdict
        total += points[verdict]
    return per_q, round(total / len(golden), 2) if golden else 0.0


def evaluation_payload(
    verdicts: Mapping[str, str],
    score: float,
    results: Mapping[str, RunResult],
) -> dict[str, Any]:
    """Charge utile JSON exportable (download Streamlit / artefact de rendu)."""
    return {
        "score": score,
        "verdicts": dict(verdicts),
        "runs": {qid: result.to_dict() for qid, result in results.items()},
    }


def evaluation_json(verdicts: Mapping[str, str], score: float, results: Mapping[str, RunResult]) -> str:
    return json.dumps(evaluation_payload(verdicts, score, results), ensure_ascii=False, indent=2)


# --- Statut des clés ---------------------------------------------------------
def key_status() -> dict[str, Any]:
    """Vérifie les clés par préfixe/longueur, sans jamais afficher leur valeur.

    Heuristique volontairement simple (pas d'appel réseau) : elle détecte une clé
    absente ou un placeholder, pas une clé révoquée.
    """
    groq = os.environ.get("GROQ_API_KEY", "")
    tavily = os.environ.get("TAVILY_API_KEY", "")
    return {
        "groq_ok": groq.startswith("gsk_") and len(groq) > 30,
        "tavily_ok": tavily.startswith("tvly-") and len(tavily) > 20,
        "model": DEFAULT_MODEL,
    }


# --- Pré-vol / diagnostic ----------------------------------------------------
# Une seule source de vérité, partagée par `check.py` (CLI) et le panneau
# « Diagnostic » de la barre latérale Streamlit : un contrôle n'est jamais
# dupliqué, donc les deux affichages ne peuvent pas diverger.
#
# Convention de criticité :
#   critical=True  → bloque l'usage de l'agent (❌)
#   critical=False → simple avertissement (⚠️)
REQUIRED_PACKAGES: dict[str, str] = {
    "streamlit": "streamlit",
    "langchain": "langchain",
    "langchain_core": "langchain-core",
    "langchain_groq": "langchain-groq",
    "langchain_chroma": "langchain-chroma",
    "langchain_huggingface": "langchain-huggingface",
    "langchain_tavily": "langchain-tavily",
    "langchain_text_splitters": "langchain-text-splitters",
    "chromadb": "chromadb",
    "sentence_transformers": "sentence-transformers",
    "tiktoken": "tiktoken",
    "arxiv": "arxiv",
    "tavily": "tavily-python",
    "dotenv": "python-dotenv",
}


@dataclass
class CheckItem:
    """Ligne de diagnostic : un point vérifié, son verdict et la correction."""

    group: str
    label: str
    ok: bool
    detail: str = ""
    hint: str = ""
    critical: bool = True

    @property
    def icon(self) -> str:
        if self.ok:
            return "✅"
        return "❌" if self.critical else "⚠️"

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "label": self.label,
            "ok": self.ok,
            "detail": self.detail,
            "hint": self.hint,
            "critical": self.critical,
        }


def _module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):  # paquet parent cassé / nom invalide
        return False


def _parse_env_file(path: str) -> dict[str, str]:
    """Lecture minimale d'un fichier .env (KEY=value), sans effet de bord."""
    values: dict[str, str] = {}
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip().strip('"').strip("'")
    except OSError:
        return {}
    return values


def _looks_like_real_key(name: str, value: str) -> bool:
    """Une vraie clé, par opposition aux placeholders livrés dans .env.example."""
    if not value or "votre" in value.lower() or "your" in value.lower():
        return False
    if value.endswith("_ICI") or value.endswith("_HERE"):
        return False
    if name == "GROQ_API_KEY":
        return value.startswith("gsk_") and len(value) > 30
    if name == "TAVILY_API_KEY":
        return value.startswith("tvly-") and len(value) > 20
    return False


def check_env_file() -> list[CheckItem]:
    """Le fichier `.env` existe-t-il, et les clés ne sont-elles pas au mauvais endroit ?

    C'est le piège n°1 : coller sa clé dans `.env.example`, que rien ne lit et qui
    n'est pas ignoré par git.
    """
    items: list[CheckItem] = []

    if ENV_PATH:
        items.append(CheckItem("Configuration", ENV_FILE, True, f"chargé ({env_source()})"))
    else:
        items.append(
            CheckItem(
                "Configuration",
                ENV_FILE,
                False,
                "introuvable",
                f"copier {ENV_EXAMPLE_FILE} vers {ENV_FILE} "
                f"(Copy-Item {ENV_EXAMPLE_FILE} {ENV_FILE}) puis y coller les clés",
            )
        )

    example_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ENV_EXAMPLE_FILE)
    exposed = [
        name
        for name, value in _parse_env_file(example_path).items()
        if _looks_like_real_key(name, value)
    ]
    if exposed:
        items.append(
            CheckItem(
                "Configuration",
                f"clés réelles dans {ENV_EXAMPLE_FILE}",
                False,
                f"{', '.join(sorted(exposed))} — fichier versionné, non ignoré par git",
                f"déplacer ces clés dans {ENV_FILE} puis remettre les placeholders, "
                "et régénérer les clés exposées",
                critical=False,
            )
        )

    return items


def check_packages() -> list[CheckItem]:
    """Présence des paquets requis (aucun import lourd, aucun réseau).

    `find_spec` vérifie qu'un module est importable sans l'importer : le diagnostic
    reste instantané même quand torch n'est pas encore chargé.
    """
    items: list[CheckItem] = []
    for module, package in REQUIRED_PACKAGES.items():
        present = _module_available(module)
        items.append(
            CheckItem(
                group="Paquets",
                label=package,
                ok=present,
                detail="" if present else "absent",
                hint="" if present else "uv sync",
            )
        )
    return items


def check_keys(groq_key: str | None = None, tavily_key: str | None = None) -> list[CheckItem]:
    """Forme des clés API (préfixe + longueur), sans jamais les afficher.

    `groq_key` / `tavily_key` permettent de tester une clé saisie dans l'UI sans
    l'écrire dans l'environnement du processus.
    """
    groq = os.environ.get("GROQ_API_KEY", "") if groq_key is None else groq_key
    tavily = os.environ.get("TAVILY_API_KEY", "") if tavily_key is None else tavily_key

    groq_ok = groq.startswith("gsk_") and len(groq) > 30
    tavily_ok = tavily.startswith("tvly-") and len(tavily) > 20

    if groq_ok:
        groq_item = CheckItem("Clés API", "GROQ_API_KEY", True, "ok")
    else:
        groq_item = CheckItem(
            "Clés API",
            "GROQ_API_KEY",
            False,
            "absente" if not groq else "format inattendu",
            "ajouter la clé dans .env (préfixe gsk_)"
            if ENV_PATH
            else f"commencer par créer {ENV_FILE} : Copy-Item {ENV_EXAMPLE_FILE} {ENV_FILE}",
        )

    if tavily_ok:
        tavily_item = CheckItem("Clés API", "TAVILY_API_KEY", True, "ok")
    else:
        tavily_item = CheckItem(
            "Clés API",
            "TAVILY_API_KEY",
            False,
            "absente — outil web désactivé" if not tavily else "format inattendu",
            "optionnel : sans clé, tavily_web_search devient un stub",
            critical=False,
        )

    return [groq_item, tavily_item]


def check_corpus() -> CheckItem:
    """Découpage du corpus (exercice réel du splitter, sans embeddings)."""
    try:
        chunks = split_corpus()
    except Exception as exc:
        return CheckItem("Corpus", "Découpage", False, str(exc), "uv sync")
    sources = sorted({doc.metadata.get("source", "?") for doc in chunks})
    complete = len(sources) == len(CORPUS)
    return CheckItem(
        group="Corpus",
        label="Découpage",
        ok=complete,
        detail=f"{len(CORPUS)} documents → {len(chunks)} chunks · {', '.join(sources)}",
        hint="" if complete else "vérifier CORPUS dans agent_core.py",
    )


def check_embedding_cache() -> CheckItem:
    """Le modèle d'embedding est-il déjà téléchargé ? (purement informatif)"""
    repo = f"sentence-transformers/{EMBEDDING_MODEL}"
    cached = False
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = isinstance(try_to_load_from_cache(repo, "config.json"), str)
    except Exception:
        pass
    return CheckItem(
        group="Index",
        label=EMBEDDING_MODEL,
        ok=cached,
        detail="en cache local" if cached else "téléchargé au premier usage (~90 Mo)",
        hint="" if cached else "nécessite une connexion au premier lancement",
        critical=False,
    )


def preflight(
    *,
    include_corpus: bool = True,
    groq_key: str | None = None,
    tavily_key: str | None = None,
) -> list[CheckItem]:
    """Toutes les vérifications, dans l'ordre d'affichage."""
    items: list[CheckItem] = []
    items.extend(check_env_file())
    items.extend(check_packages())
    items.extend(check_keys(groq_key, tavily_key))
    if include_corpus:
        items.append(check_corpus())
    items.append(check_embedding_cache())
    return items


def preflight_ready(items: Iterable[CheckItem]) -> bool:
    """Prêt à interroger l'agent ? (aucun point bloquant en échec)"""
    return not any(item.critical and not item.ok for item in items)

