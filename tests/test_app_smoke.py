"""Test de fumée de l'interface Streamlit (rendu initial, sans appel réseau).

Ces tests n'utilisent ni Groq, ni Tavily, ni le modèle d'embedding : ils vérifient
que l'application se construit, expose les bons contrôles et échoue proprement
quand une dépendance manque.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _app() -> AppTest:
    at = AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()
    return at


def _by_label(elements, label: str):
    return next(element for element in elements if element.label == label)


def test_app_renders_without_exception() -> None:
    at = _app()
    assert not list(at.exception)


def test_app_exposes_four_tabs() -> None:
    at = _app()
    assert [tab.label for tab in at.tabs] == [
        "💬 Agent",
        "📊 Golden set",
        "🔍 Trace",
        "📚 Corpus",
    ]


def test_sidebar_contains_the_preflight_diagnostic() -> None:
    at = _app()
    assert any(check.label == "Afficher aussi les points OK" for check in at.checkbox)
    # une ligne de synthèse par groupe (les points OK ne s'affichent que sur demande)
    assert any(caption.value.startswith("**Paquets**") for caption in at.caption)
    assert any(caption.value.startswith("**Clés API**") for caption in at.caption)

    at.checkbox[0].check().run()
    assert any("`streamlit`" in markdown.value for markdown in at.markdown)
    assert any("`langchain-groq`" in markdown.value for markdown in at.markdown)


def test_corpus_metrics_are_computed_without_embeddings() -> None:
    at = _app()
    metrics = {metric.label: metric.value for metric in at.metric}
    assert metrics["Documents"] == "4"
    assert int(metrics["Chunks"]) >= 4
    assert metrics["Embedding"] == "all-MiniLM-L6-v2"


def test_chat_is_disabled_without_groq_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    at = _app()
    ask = _by_label(at.button, "▶️ Interroger l'agent")
    assert ask.disabled is True
    assert any("GROQ_API_KEY" in warning.value for warning in at.warning)


def test_retriever_search_never_raises() -> None:
    """Avec ou sans dépendances lourdes, l'UI ne doit jamais planter.

    - paquets présents : la recherche renvoie des extraits, aucune erreur ;
    - paquets absents / hors ligne : une erreur explicite, mais pas d'exception.
    """
    at = _app()
    _by_label(at.text_input, "Requête de test").set_value("How does HNSW work?").run()
    _by_label(at.button, "🔎 Rechercher").click().run()

    assert not list(at.exception)

    index_errors = [error.value for error in at.error if "index vectoriel" in error.value]
    if index_errors:
        # dépendances absentes / hors ligne : message explicite, pas de crash
        assert "index vectoriel" in index_errors[0]
    else:
        # index construit : le retriever affiche des extraits avec leur distance
        assert any("distance" in expander.label for expander in at.expander)
