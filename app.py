"""Interface Streamlit du ReAct agent — TP 4.a (Partie 1).

Ce fichier ne contient **que de la présentation** : widgets, mise en page, caches et
état de session. Toute la logique (corpus, index, outils, agent, traces, score,
pré-vol) vit dans `agent_core.py`, qui n'importe jamais Streamlit.

Lancement :
    uv run streamlit run app.py       # http://localhost:8501

Plan du fichier (dans l'ordre d'exécution Streamlit) :
    1. Configuration de la page
    2. Résolution des clés API (.env / secrets.toml, jamais saisies dans l'UI)
    3. Ressources coûteuses mises en cache (embeddings, index, agent)
    4. État de session (conversation, évaluation, dernier run)
    5. Panneau de diagnostic (mêmes contrôles que check.py)
    6. Barre latérale de configuration
    7. Quatre onglets : Agent, Golden set, Trace, Corpus

Rappel Streamlit : **tout le script est ré-exécuté** à chaque interaction. D'où les
caches (`st.cache_resource` / `st.cache_data`) et `st.session_state` : sans eux, le
modèle d'embedding serait rechargé et la conversation perdue à chaque clic.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any

import streamlit as st

import agent_core as core

# ---------------------------------------------------------------------------
# 1. Configuration de la page (doit être le premier appel Streamlit)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TP 4.a — ReAct Agent",
    page_icon="🤖",
    layout="wide",  # largeur utile pour les tableaux et la trace
    initial_sidebar_state="expanded",
)

# Libellés affichés pour les trois verdicts du score de routage.
VERDICT_LABEL = {"exact": "✅ exact", "partiel": "≈ partiel", "faux": "❌ faux"}


# ---------------------------------------------------------------------------
# 2. Clés API : lues dans .env / .streamlit/secrets.toml (jamais saisies dans l'UI)
# ---------------------------------------------------------------------------
def resolve_key(name: str) -> str:
    """Clé effective pour `name`, sans jamais l'afficher.

    Priorité : `.streamlit/secrets.toml` (déploiement) puis l'environnement, que
    `agent_core` a rempli depuis `.env` au moment de son import.
    """
    try:
        if name in st.secrets:  # lève une exception s'il n'y a pas de secrets.toml
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return os.environ.get(name, "").strip()


# ---------------------------------------------------------------------------
# 3. Ressources coûteuses (partagées entre sessions, chargées à la demande)
#
# `cache_resource` = un seul objet partagé par tous les utilisateurs/sessions.
# À ne pas confondre avec `cache_data` (copie sérialisée, pour les données).
# Le préfixe `_` d'un argument (`_embeddings`) dit à Streamlit de ne pas le hasher :
# un objet HuggingFaceEmbeddings n'est ni hashable ni sérialisable.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Chargement du modèle d'embedding (all-MiniLM-L6-v2)…")
def get_embeddings() -> Any:
    """Charge all-MiniLM-L6-v2 (~90 Mo au premier lancement, puis cache disque HF).

    L'import est local : la page s'affiche sans attendre torch/sentence-transformers.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=core.EMBEDDING_MODEL)


@st.cache_resource(show_spinner="Indexation du corpus dans Chroma…")
def get_store(_embeddings: Any) -> tuple[Any, list[Any]]:
    """Construit l'index Chroma en mémoire (4 documents → quelques chunks).

    Pas de `persist_directory` : l'index repart de zéro à chaque démarrage à froid,
    ce qui évite l'empilement de chunks (4 → 8 → 12) constaté dans le notebook.
    """
    return core.build_vectorstore(_embeddings)


@st.cache_resource(show_spinner="Création de l'agent ReAct…")
def get_agent(
    model: str,
    temperature: float,
    max_tokens: int,
    effort: str,
    enable_web: bool,
    enable_arxiv: bool,
    system_prompt: str,
    groq_key: str,
    tavily_key: str,
) -> tuple[Any, list[Any]]:
    """Compile l'agent et renvoie (agent, outils).

    Astuce de conception : **les arguments sont la clé de cache**. Changer le modèle,
    la température, un outil ou le prompt crée donc un nouvel agent au lieu de
    réutiliser un agent obsolète (piège documenté dans le notebook d'origine).
    """
    llm = core.build_llm(
        model=model,
        api_key=groq_key,
        temperature=temperature,
        max_tokens=max_tokens,
        # "désactivé" → on n'envoie pas le paramètre (modèles Groq non « reasoning »).
        reasoning_effort=None if effort == "désactivé" else effort,
    )

    # Le retriever est reconstruit à partir de l'index déjà en cache (coût quasi nul).
    store, _chunks = get_store(get_embeddings())
    tools = core.build_tools(
        retriever=core.build_retriever(store),
        tavily_key=tavily_key,
        enable_web=enable_web,
        enable_arxiv=enable_arxiv,
    )
    return core.build_agent(llm, tools, system_prompt), tools


@st.cache_data(show_spinner=False)
def corpus_chunks() -> list[dict[str, str]]:
    """Découpage du corpus pour l'onglet Corpus — sans embeddings.

    `cache_data` (et non `cache_resource`) : on renvoie des dictionnaires simples,
    sérialisables, donc Streamlit peut en garder une copie.
    """
    return [
        {"source": doc.metadata.get("source", "?"), "content": doc.page_content}
        for doc in core.split_corpus()
    ]


def load_store() -> tuple[Any, list[Any]]:
    """Accès à l'index avec un message d'erreur lisible plutôt qu'une stacktrace."""
    try:
        return get_store(get_embeddings())
    except Exception as exc:
        st.error(f"Impossible de construire l'index vectoriel : {exc}")
        st.caption(
            "Le premier lancement télécharge `all-MiniLM-L6-v2` (~90 Mo). "
            "Vérifie la connexion réseau et que `sentence-transformers` est installé."
        )
        st.stop()  # interrompt proprement ce rerun (pas d'exception dans l'UI)


# ---------------------------------------------------------------------------
# 4. État de session (survit aux reruns)
#     history     : liste de tours {question, result} — tout l'historique du chat
#     evaluation  : dernier golden set {results, verdicts, score, date, model}
#     last_result : (source, libellé, RunResult) affiché dans l'onglet Trace
# ---------------------------------------------------------------------------
st.session_state.setdefault("history", [])
st.session_state.setdefault("evaluation", None)
st.session_state.setdefault("last_result", None)


# ---------------------------------------------------------------------------
# 5. Panneau de diagnostic : mêmes vérifications que `check.py`
# ---------------------------------------------------------------------------
def render_diagnostic(groq_key: str = "", tavily_key: str = "") -> None:
    """Affiche `agent_core.preflight()` dans la barre latérale.

    Volontairement **non caché** : créer `.env` ou installer un paquet doit changer
    le verdict au rerun suivant, sans redémarrer l'application.
    """
    items = core.preflight(
        # `or None` → « lis l'environnement » si la valeur est vide.
        groq_key=groq_key or None,
        tavily_key=tavily_key or None,
    )
    ready = core.preflight_ready(items)
    blocking = [item for item in items if item.critical and not item.ok]

    # Bandeau de synthèse : vert si aucun point bloquant, rouge sinon.
    if ready:
        st.success("Configuration prête.", icon="🟢")
    else:
        st.error(f"{len(blocking)} point(s) bloquant(s) à corriger.", icon="🔴")

    # Par défaut, on n'encombre pas la sidebar : seuls les problèmes sont listés.
    show_all = st.checkbox("Afficher aussi les points OK", value=False, key="diag_show_all")

    # Regroupement par thème (Configuration, Paquets, Clés API, Corpus, Index).
    groups: dict[str, list[core.CheckItem]] = {}
    for item in items:
        groups.setdefault(item.group, []).append(item)

    for group, group_items in groups.items():
        visible = [item for item in group_items if show_all or not item.ok]
        if not visible:
            # Tout va bien dans ce groupe : une simple ligne de synthèse suffit.
            ok_count = len(group_items)
            st.caption(f"**{group}** — {ok_count}/{ok_count} ✅")
            continue

        ok_count = len([item for item in group_items if item.ok])
        st.caption(f"**{group}** — {ok_count}/{len(group_items)} OK")
        for item in visible:
            # ✅ / ❌ (bloquant) / ⚠️ (optionnel) sont décidés par CheckItem.icon
            line = f"{item.icon} `{item.label}`"
            if item.detail:
                line += f" — {item.detail}"
            st.markdown(line)
            if not item.ok and item.hint:
                st.caption(f"→ {item.hint}")  # la correction, en petit

    # Un bouton explicite pour relancer après avoir créé .env ou installé un paquet.
    if st.button("🔄 Relancer les vérifications", key="diag_reload"):
        st.rerun()


# ---------------------------------------------------------------------------
# 6. Barre latérale : tout ce qui influe sur un run
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🤖 TP 4.a — ReAct")
    st.caption("Agent de recherche · LangChain 1.x · Groq")

    # Lecture des clés + verdict de format (mêmes règles que le pré-vol).
    groq_key = resolve_key("GROQ_API_KEY")
    tavily_key = resolve_key("TAVILY_API_KEY")
    groq_ok = groq_key.startswith("gsk_") and len(groq_key) > 30
    tavily_ok = tavily_key.startswith("tvly-") and len(tavily_key) > 20

    # --- Statut des clés : lecture seule, aucune saisie (rien ne fuit dans l'UI) ---
    with st.expander("🔑 Clés API", expanded=not groq_ok):
        st.markdown(f"{'✅' if groq_ok else '❌'} **Groq** — {'détectée' if groq_ok else 'absente'}")
        st.markdown(
            f"{'✅' if tavily_ok else '⚠️'} **Tavily** — "
            f"{'détectée' if tavily_ok else 'absente (outil web désactivé)'}"
        )
        st.caption(f"Source : `{core.env_source()}`")  # ex. « .env », jamais le contenu

        # Relire .env sans redémarrer : indispensable après avoir créé le fichier.
        if st.button("♻️ Recharger .env", key="reload_env"):
            core.reload_env()
            st.cache_resource.clear()  # l'agent en cache porte l'ancienne clé
            st.rerun()

        st.caption("Les clés se lisent dans `.env` ou `.streamlit/secrets.toml` — rien à saisir ici.")

    # --- Diagnostic partagé avec check.py ---
    with st.expander("🩺 Diagnostic", expanded=not groq_ok):
        render_diagnostic(groq_key, tavily_key)

    # --- Réglages du LLM ---
    st.subheader("Modèle")
    model = st.text_input(
        "Modèle Groq",
        value=core.DEFAULT_MODEL,
        help="`llama-3.3-70b-versatile` renvoie 404 sur le compte du cours.",
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05)  # 0 par défaut : le plus stable
    col_a, col_b = st.columns(2)
    max_tokens = col_a.number_input("max_tokens", 256, 4096, 1024, 64)
    effort = col_b.selectbox("reasoning_effort", ["low", "medium", "high", "désactivé"], index=0)
    recursion_limit = st.slider(
        "recursion_limit",
        4,
        24,
        core.DEFAULT_RECURSION_LIMIT,
        help="Garde-fou anti-boucle : nombre maximal d'itérations de l'agent.",
    )

    # --- Outils exposés au LLM ---
    st.subheader("Outils")
    # Tavily grisé sans clé : le stub répondrait « désactivé » pour rien.
    enable_web = st.toggle("🌐 tavily_web_search", value=tavily_ok, disabled=not tavily_ok)
    enable_arxiv = st.toggle("📄 arxiv_search", value=True)
    if not tavily_ok:
        enable_web = False  # ceinture et bretelles : un toggle désactivé garde sa valeur

    # --- Réglages avancés (prompt, quota, caches) ---
    with st.expander("⚙️ Avancé"):
        system_prompt = st.text_area(
            "Prompt système",
            value=core.SYSTEM_PROMPT,
            height=260,
            help="Seule politique de routage d'un agent ReAct : le LLM n'est pas contraint mécaniquement.",
        )
        retry_delay = st.number_input(
            "Pause avant reprise (s)",
            0,
            300,
            60,
            10,
            help="Limite gratuite Groq : 8 000 TPM sur gpt-oss-20b.",
        )
        st.caption("Reconstruire l'agent est nécessaire après un changement de prompt/outils : c'est automatique ici.")

        # Purge totale : ressources (embeddings/index/agent) + données + résultats.
        if st.button("♻️ Vider les caches"):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.session_state["evaluation"] = None
            st.session_state["last_result"] = None
            st.rerun()

    st.divider()
    st.caption(
        f"Index : `{core.COLLECTION_NAME}` · k={core.DEFAULT_K} · "
        f"chunks {core.CHUNK_SIZE}/{core.CHUNK_OVERLAP}"
    )


def agent_or_none() -> Any:
    """Récupère l'agent (ou le crée) et transforme un échec en message lisible.

    Cas typiques d'échec : clé invalide, modèle inconnu, paquet manquant.
    On renvoie None plutôt que de laisser remonter l'exception : l'UI reste stable.
    """
    try:
        agent, _tools = get_agent(
            model,
            float(temperature),
            int(max_tokens),
            effort,
            enable_web,
            enable_arxiv,
            system_prompt,
            groq_key,
            tavily_key,
        )
        return agent
    except Exception as exc:
        st.error(f"L'agent n'a pas pu être créé : {exc}")
        return None


def retry_notice(delay: float) -> None:
    """Callback appelé par le cœur avant une reprise sur limite TPM.

    On affiche l'avertissement AVANT de dormir : sinon l'utilisateur voit une page
    figée pendant 60 s sans explication.
    """
    st.warning(f"⏳ Limite TPM atteinte — pause de {delay:.0f} s puis un nouvel essai.")
    time.sleep(delay)


# ---------------------------------------------------------------------------
# 7. Rendu de la trace et des métriques (briques d'affichage réutilisées)
# ---------------------------------------------------------------------------
def render_trace(trace: list[core.TraceStep], *, max_observation: int = 900) -> None:
    """Rend une trace ReAct : Action → Observation → Thought → Réponse finale.

    `max_observation` tronque les observations : les extraits du corpus peuvent être
    longs et noieraient la réponse dans l'onglet.
    """
    if not trace:
        st.caption("Trace vide.")
        return

    for step in trace:
        if step.kind == "action":
            # Arguments sérialisés en JSON : on voit la requête réellement envoyée.
            args = json.dumps(step.args, ensure_ascii=False)
            st.markdown(f"**{step.icon} Action** · `{step.name}({args})`")

        elif step.kind == "observation":
            body = step.content[:max_observation]
            if len(step.content) > max_observation:
                body += "\n…"
            st.markdown(f"**{step.icon} Observation** · `{step.name}`")
            st.code(body, language="text")  # bloc monospace = lisible et copiable

        elif step.kind == "thought":
            # Un « Thought » n'apparaît que si le modèle écrit du texte ET appelle un outil.
            st.markdown(f"**{step.icon} Thought**")
            st.markdown(step.content)

        elif step.kind == "final":
            st.markdown(f"**{step.icon} Réponse finale**")
            st.markdown(step.content)

        else:  # question de l'utilisateur
            st.markdown(f"**{step.icon} {step.label}** · {step.content}")


def render_metrics(result: core.RunResult) -> None:
    """Quatre métriques par run : outils, étapes, latence, reprise."""
    cols = st.columns(4)
    cols[0].metric("Outils", ", ".join(result.tools_used) or "aucun")
    cols[1].metric("Étapes", result.steps)  # 1 étape = 1 appel d'outil
    cols[2].metric("Durée", f"{result.duration_s:.2f} s")
    cols[3].metric("Reprise TPM", "oui" if result.retried else "non")


# ---------------------------------------------------------------------------
# 8. Les quatre onglets
# ---------------------------------------------------------------------------
tab_chat, tab_eval, tab_trace, tab_corpus = st.tabs(
    ["💬 Agent", "📊 Golden set", "🔍 Trace", "📚 Corpus"]
)

# ---------------------------------------------------------------------- Chat
with tab_chat:
    head_l, head_r = st.columns([3, 1])
    with head_l:
        st.subheader("Pose une question à l'agent")
        st.caption(
            "L'agent choisit lui-même entre corpus local, web et arXiv. "
            "Les étapes Thought → Action → Observation s'affichent en direct."
        )
    with head_r:
        # Vider l'historique = vider le session_state (pas de variable globale).
        if st.session_state["history"] and st.button("🧹 Vider la conversation"):
            st.session_state["history"] = []
            st.rerun()

    if not groq_ok:
        st.warning(
            "Ajoute `GROQ_API_KEY` dans `.env` (fichier ignoré par git), "
            "puis clique sur ♻️ **Recharger .env** dans la barre latérale."
        )

    # --- Historique : rendu depuis le session_state à chaque rerun ---
    for turn in st.session_state["history"]:
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(turn["question"])

        result: core.RunResult = turn["result"]
        with st.chat_message("assistant", avatar="🤖"):
            if result.error:
                st.error(result.answer)  # l'erreur reste visible, pas avalée
            else:
                st.markdown(result.answer or "_Réponse vide — inspecte la trace._")
            render_metrics(result)
            with st.expander("🔍 Trace ReAct"):
                render_trace(result.trace)

    # --- Saisie : un `form` évite un rerun à chaque frappe ---
    with st.form("ask_form", clear_on_submit=True):
        question = st.text_area(
            "Question",
            height=80,
            placeholder="Ex. : Comment fonctionne HNSW pour la recherche de plus proches voisins ?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button(
            "▶️ Interroger l'agent", type="primary", disabled=not groq_ok
        )

    if submitted and question.strip():
        agent = agent_or_none()
        if agent is not None:
            # `st.status` = conteneur qu'on peut mettre à jour pendant l'exécution :
            # c'est ce qui donne la sensation de « boucle ReAct en direct ».
            with st.status("Boucle ReAct en cours…", expanded=True) as status:
                seen: list[str] = []

                def on_step(step: core.TraceStep) -> None:
                    """Callback de streaming : appelé pour chaque étape produite."""
                    if step.kind == "action":
                        seen.append(step.name or "?")
                        status.write(
                            f"🤖 **Action** `{step.name}` {json.dumps(step.args, ensure_ascii=False)}"
                        )
                    elif step.kind == "observation":
                        preview = step.content[:120].replace("\n", " ")  # 1 ligne
                        status.write(f"📥 **Observation** `{step.name}` — {preview}…")
                    elif step.kind == "final":
                        status.write("💬 **Réponse finale** reçue")

                result = core.run_react_agent(
                    agent,
                    question.strip(),
                    recursion_limit=int(recursion_limit),
                    on_step=on_step,  # mode streaming
                    retry_on_rate_limit=True,
                    retry_delay=float(retry_delay),
                    on_retry=retry_notice,
                )

                # Bandeau final : résumé en une ligne, replié pour laisser la place.
                status.update(
                    label=(
                        f"Terminé · {len(result.tools_used)} outil(s) · {result.duration_s:.2f} s"
                        if not result.error
                        else "Échec du run"
                    ),
                    state="error" if result.error else "complete",
                    expanded=False,
                )

            # On mémorise le tour puis on relance le script : l'historique (avec la
            # trace) est réaffiché au-dessus du formulaire, dans le bon ordre.
            st.session_state["history"].append({"question": question.strip(), "result": result})
            st.session_state["last_result"] = ("chat", question.strip(), result)
            st.rerun()

# ---------------------------------------------------------------- Golden set
with tab_eval:
    st.subheader("Golden set — évaluation du routage")
    st.caption(
        "5 questions + outils attendus. Le score mesure **le routage uniquement** : "
        "une réponse peut être exacte en outils et fausse en contenu (cf. Q2/Q3 du rapport)."
    )

    # Tableau de référence : ce qui est attendu, avant toute exécution.
    st.dataframe(
        [
            {
                "Q": item["id"],
                "Question": item["question"],
                "Outils attendus": ", ".join(item["expected_tools"]) or "aucun (refus)",
            }
            for item in core.GOLDEN_SET
        ],
        hide_index=True,
    )

    col_run, col_info = st.columns([1, 2])
    run_eval = col_run.button(
        "▶️ Lancer l'évaluation",
        type="primary",
        disabled=not groq_ok,  # inutile de cliquer sans clé
    )
    col_info.caption(
        "Coût indicatif : 5 appels LLM (2 à 20 s chacun). Une limite TPM déclenche "
        f"une pause de {int(retry_delay)} s et une seule reprise par question."
    )

    if run_eval:
        agent = agent_or_none()
        if agent is not None:
            results: dict[str, core.RunResult] = {}
            progress = st.progress(0.0, text="Démarrage…")

            # Exécution séquentielle : le tier gratuit ne supporte pas 5 appels
            # simultanés (8 000 TPM), et l'ordre des questions doit rester lisible.
            for index, item in enumerate(core.GOLDEN_SET):
                progress.progress(
                    index / len(core.GOLDEN_SET),
                    text=f"{item['id']} — {item['question'][:70]}…",
                )
                results[item["id"]] = core.run_react_agent(
                    agent,
                    item["question"],
                    recursion_limit=int(recursion_limit),
                    retry_on_rate_limit=True,
                    retry_delay=float(retry_delay),
                    on_retry=retry_notice,
                )
            progress.progress(1.0, text="Évaluation terminée")

            # Score calculé par le cœur (testé unitairement, 0.80 sur la réf.).
            verdicts, score = core.routing_score(core.GOLDEN_SET, results)
            st.session_state["evaluation"] = {
                "results": results,
                "verdicts": verdicts,
                "score": score,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model": model,
            }

            # La trace consultable devient la dernière question du golden set.
            last = results[core.GOLDEN_SET[-1]["id"]]
            st.session_state["last_result"] = ("eval", "Dernière question du golden set", last)

    # --- Résultats persistants (relus depuis le session_state) ---
    evaluation = st.session_state["evaluation"]
    if not evaluation:
        st.info("Aucune évaluation en mémoire : lance-la pour obtenir un score de routage.")
    else:
        st.divider()
        left, right = st.columns([1, 2])
        left.metric("Score de routage", f"{evaluation['score']:.2f} / 1.00")
        right.progress(evaluation["score"])  # barre proportionnelle au score
        st.caption(
            f"Run du {evaluation['date']} · modèle `{evaluation['model']}` · `temperature={temperature}`"
        )

        # Tableau Attendu / Observé : la lecture la plus rapide du routage.
        rows = []
        for item in core.GOLDEN_SET:
            result = evaluation["results"][item["id"]]
            verdict = evaluation["verdicts"][item["id"]]
            rows.append(
                {
                    "Q": item["id"],
                    "Attendu": ", ".join(item["expected_tools"]) or "aucun",
                    "Observé": ", ".join(result.tools_used) or "aucun",
                    "Étapes": result.steps,
                    "Durée (s)": result.duration_s,
                    "Verdict": VERDICT_LABEL.get(verdict, verdict),
                }
            )
        st.dataframe(rows, hide_index=True)

        # Détail par question : réponse, métriques et trace complète.
        for item in core.GOLDEN_SET:
            result = evaluation["results"][item["id"]]
            verdict = evaluation["verdicts"][item["id"]]
            with st.expander(
                f"{VERDICT_LABEL.get(verdict, verdict)} · {item['id']} — {item['question']}"
            ):
                st.markdown(result.answer or "_Réponse vide._")
                render_metrics(result)
                render_trace(result.trace)

        # Export pour le rapport : trace + verdicts + score dans un seul JSON.
        st.download_button(
            "⬇️ Exporter le run (JSON)",
            data=core.evaluation_json(
                evaluation["verdicts"], evaluation["score"], evaluation["results"]
            ),
            file_name=f"golden_set_{datetime.now():%Y%m%d_%H%M}.json",
            mime="application/json",
        )
        st.caption(
            "Les chiffres publiés dans le README du notebook (score 0.8) ne sont pas "
            "reproductibles à l'identique : à temperature 0, le choix des outils varie encore."
        )

# --------------------------------------------------------------------- Trace
with tab_trace:
    st.subheader("Trace structurée du dernier run")
    last_result = st.session_state["last_result"]

    if not last_result:
        st.info("Lance une question (onglet Agent) ou une évaluation (onglet Golden set).")
    else:
        source, label, result = last_result
        st.caption(f"Source : **{source}** · {label}")
        render_metrics(result)
        render_trace(result.trace)

        # JSON brut : ce que contient réellement un RunResult (utile pour déboguer).
        with st.expander("🧾 JSON brut"):
            st.json(result.to_dict())

        st.download_button(
            "⬇️ Télécharger ce run (JSON)",
            data=json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            file_name="react_run.json",
            mime="application/json",
        )

# -------------------------------------------------------------------- Corpus
with tab_corpus:
    st.subheader("Corpus local indexé")

    # `corpus_chunks` est caché et ne charge aucun embedding : l'onglet s'affiche
    # même si le modèle n'est pas encore téléchargé.
    chunks = corpus_chunks()
    metrics = st.columns(4)
    metrics[0].metric("Documents", len(core.CORPUS))
    metrics[1].metric("Chunks", len(chunks))
    metrics[2].metric("Embedding", core.EMBEDDING_MODEL)
    metrics[3].metric("Top-k", core.DEFAULT_K)

    st.caption(
        f"Chunks de ~{core.CHUNK_SIZE} tokens (overlap {core.CHUNK_OVERLAP}) découpés avec "
        "`RecursiveCharacterTextSplitter`. Avec k=4 et 4 documents, tout le corpus remonte : "
        "seul l'ordre varie, la pertinence du retrieval n'est donc pas mesurable ici."
    )

    # Reconstruction explicite : utile après avoir modifié le corpus.
    if st.button("🔨 Construire / recharger l'index"):
        get_store.clear()
        load_store()
        st.success("Index reconstruit.")

    st.divider()
    st.markdown("#### Tester le retriever")

    # Pas de formulaire ici : deux widgets suffisent, et le bouton n'exécute la
    # recherche que lorsqu'il est cliqué (donc pas de chargement du modèle au rendu).
    query = st.text_input("Requête de test", placeholder="How does HNSW work?")
    if st.button("🔎 Rechercher") and query.strip():
        store, _ = load_store()
        try:
            # `similarity_search_with_score` renvoie (Document, distance).
            hits = store.similarity_search_with_score(query, k=core.DEFAULT_K)
        except Exception as exc:
            st.error(f"Recherche impossible : {exc}")
        else:
            for doc, distance in hits:
                with st.expander(
                    f"{doc.metadata.get('source', '?')} · distance {distance:.4f} (plus petit = plus proche)"
                ):
                    st.code(doc.page_content, language="markdown")

    st.divider()
    st.markdown("#### Documents sources")

    # Le corpus est petit : on peut l'afficher intégralement, sans pagination.
    for name, text in core.CORPUS.items():
        with st.expander(name):
            st.code(text, language="markdown")

st.divider()
st.caption(
    "Projet pédagogique — TP 4.a, Module 4 « AI Agents ». "
    "Un agent ReAct pur n'a aucun mécanisme d'auto-évaluation : c'est ce qu'ajoute la Partie 3 (LangGraph)."
)
