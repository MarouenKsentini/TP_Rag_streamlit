"""Tests du cœur métier (aucun appel réseau, aucun modèle téléchargé)."""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import agent_core as core


# --- Corpus / découpage ------------------------------------------------------
def test_corpus_contains_four_named_documents() -> None:
    assert len(core.CORPUS) == 4
    assert set(core.CORPUS) == {
        "transformers.md",
        "rag_systems.md",
        "vector_databases.md",
        "finetuning.md",
    }


def test_split_corpus_keeps_sources_without_tiktoken() -> None:
    pytest.importorskip("langchain_text_splitters")
    chunks = core.split_corpus(use_tiktoken=False)
    assert chunks, "le corpus doit produire au moins un chunk"
    assert {doc.metadata["source"] for doc in chunks} == set(core.CORPUS)
    assert all(doc.page_content for doc in chunks)


def test_golden_set_is_stable() -> None:
    ids = [item["id"] for item in core.GOLDEN_SET]
    assert ids == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    known = {"rag_search", "tavily_web_search", "arxiv_search"}
    for item in core.GOLDEN_SET:
        assert set(item["expected_tools"]) <= known
        assert item["question"].strip()
    # Q5 doit rester un refus sans outil (test du garde-fou du prompt)
    assert core.GOLDEN_SET[-1]["expected_tools"] == []


# --- Verdicts / score --------------------------------------------------------
@pytest.mark.parametrize(
    ("expected", "observed", "verdict"),
    [
        (["rag_search"], ["rag_search"], "exact"),
        ([], [], "exact"),
        (["rag_search"], ["tavily_web_search"], "faux"),
        (["rag_search"], [], "faux"),
        ([], ["rag_search"], "faux"),  # refus attendu non respecté
        (["rag_search", "tavily_web_search"], ["tavily_web_search"], "partiel"),
        (["arxiv_search"], ["arxiv_search", "rag_search"], "partiel"),
    ],
)
def test_verdict_for(expected: list[str], observed: list[str], verdict: str) -> None:
    assert core.verdict_for(expected, observed) == verdict


def test_routing_score_reference_run_scores_0_8() -> None:
    """Reproduit la sortie de référence du notebook (Q3 et Q4 partiels)."""
    results = {
        "Q1": core.RunResult(tools_used=["rag_search"]),
        "Q2": core.RunResult(tools_used=["tavily_web_search"]),
        "Q3": core.RunResult(tools_used=["arxiv_search", "rag_search"]),
        "Q4": core.RunResult(tools_used=["tavily_web_search"]),
        "Q5": core.RunResult(tools_used=[]),
    }
    verdicts, score = core.routing_score(core.GOLDEN_SET, results)

    assert verdicts == {
        "Q1": "exact",
        "Q2": "exact",
        "Q3": "partiel",
        "Q4": "partiel",
        "Q5": "exact",
    }
    assert score == 0.8


def test_routing_score_accepts_plain_dicts() -> None:
    results = {item["id"]: {"tools_used": item["expected_tools"]} for item in core.GOLDEN_SET}
    verdicts, score = core.routing_score(core.GOLDEN_SET, results)
    assert score == 1.0
    assert set(verdicts.values()) == {"exact"}


# --- Messages -> trace -------------------------------------------------------
def test_steps_from_human_message() -> None:
    steps = core.steps_from_message(HumanMessage(content="Bonjour"))
    assert [step.kind for step in steps] == ["human"]


def test_steps_from_ai_message_with_tool_calls() -> None:
    message = AIMessage(
        content="",
        tool_calls=[{"name": "rag_search", "args": {"query": "HNSW"}, "id": "call_1"}],
    )
    steps = core.steps_from_message(message)
    assert [step.kind for step in steps] == ["action"]
    assert steps[0].name == "rag_search"
    assert steps[0].args == {"query": "HNSW"}


def test_steps_from_ai_message_thought_then_actions() -> None:
    message = AIMessage(
        content="Je vérifie le corpus local.",
        tool_calls=[
            {"name": "rag_search", "args": {"query": "LoRA"}, "id": "call_1"},
            {"name": "arxiv_search", "args": {"query": "LoRA"}, "id": "call_2"},
        ],
    )
    steps = core.steps_from_message(message)
    assert [step.kind for step in steps] == ["thought", "action", "action"]
    assert [step.name for step in steps if step.kind == "action"] == ["rag_search", "arxiv_search"]


def test_steps_from_empty_ai_message_are_ignored() -> None:
    assert core.steps_from_message(AIMessage(content="")) == []


def test_steps_from_tool_message() -> None:
    message = ToolMessage(content="extrait", name="rag_search", tool_call_id="call_1")
    steps = core.steps_from_message(message)
    assert steps[0].kind == "observation"
    assert steps[0].name == "rag_search"
    assert steps[0].content == "extrait"


def test_text_of_handles_blocks_and_none() -> None:
    assert core.text_of(None) == ""
    assert core.text_of("abc") == "abc"
    assert core.text_of([{"type": "text", "text": "a"}, "b"]) == "a\nb"


# --- Robustesse --------------------------------------------------------------
@pytest.mark.parametrize(
    "result",
    [
        core.RunResult(answer="[Erreur : Error code: 413 rate_limit_exceeded]"),
        core.RunResult(answer="", error="429 Too Many Requests"),
        core.RunResult(answer="Rate limit reached for tokens per minute"),
    ],
)
def test_is_rate_limited_detects_quota_errors(result: core.RunResult) -> None:
    assert core.is_rate_limited(result) is True


def test_is_rate_limited_ignores_normal_answers() -> None:
    assert core.is_rate_limited(core.RunResult(answer="Voici la réponse.", error=None)) is False


def test_run_react_agent_retries_once_on_rate_limit() -> None:
    """Deuxième tentative uniquement, après un on_retry explicite."""

    class FakeAgent:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, _payload, _config):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("Error code: 413 rate_limit_exceeded")
            return {"messages": [HumanMessage(content="q"), AIMessage(content="ok")]}

    agent = FakeAgent()
    waits: list[float] = []
    result = core.run_react_agent(
        agent, "question", retry_on_rate_limit=True, retry_delay=0, on_retry=waits.append
    )

    assert agent.calls == 2
    assert waits == [0]
    assert result.retried is True
    assert result.answer == "ok"
    assert result.tools_used == []


def test_run_react_agent_reports_errors_without_raising() -> None:
    class BrokenAgent:
        def invoke(self, _payload, _config):
            raise ValueError("modèle inconnu")

    result = core.run_react_agent(BrokenAgent(), "question", retry_on_rate_limit=False)
    assert result.error == "modèle inconnu"
    assert result.answer.startswith("[Erreur :")
    assert result.steps == 0


def test_run_react_agent_streams_steps() -> None:
    class StreamingAgent:
        def stream(self, _payload, _config, stream_mode="values"):
            states = [
                {"messages": [HumanMessage(content="q")]},
                {"messages": [HumanMessage(content="q"), AIMessage(
                    content="",
                    tool_calls=[{"name": "rag_search", "args": {"query": "HNSW"}, "id": "c1"}],
                )]},
                {"messages": [
                    HumanMessage(content="q"),
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "rag_search", "args": {"query": "HNSW"}, "id": "c1"}],
                    ),
                    ToolMessage(content="doc", name="rag_search", tool_call_id="c1"),
                    AIMessage(content="Réponse finale."),
                ]},
            ]
            for state in states:
                yield state

    seen: list[str] = []
    result = core.run_react_agent(
        StreamingAgent(), "q", on_step=lambda step: seen.append(step.kind), retry_on_rate_limit=False
    )

    assert seen == ["human", "action", "observation", "final"]
    assert result.tools_used == ["rag_search"]
    assert result.steps == 1
    assert result.answer == "Réponse finale."


# --- Pré-vol / diagnostic ----------------------------------------------------
def test_check_item_icons_depend_on_criticality() -> None:
    assert core.CheckItem("g", "l", True).icon == "✅"
    assert core.CheckItem("g", "l", False, critical=True).icon == "❌"
    assert core.CheckItem("g", "l", False, critical=False).icon == "⚠️"


def test_check_packages_lists_all_required_packages() -> None:
    items = core.check_packages()
    labels = {item.label for item in items}
    assert labels == set(core.REQUIRED_PACKAGES.values())
    assert all(item.group == "Paquets" for item in items)
    assert all(item.critical for item in items)
    # le détail n'expose jamais un chemin ou une version : juste « absent »
    assert all(item.detail in ("", "absent") for item in items)


def test_check_keys_uses_overrides_without_touching_environ() -> None:
    items = core.check_keys("gsk_" + "x" * 40, "")
    groq, tavily = items
    assert groq.ok is True
    assert tavily.ok is False
    assert tavily.critical is False  # l'outil web est optionnel
    assert "désactivé" in tavily.detail


@pytest.mark.parametrize(
    ("groq", "tavily", "tavily_ok"),
    [
        ("", "", False),
        ("VOTRE_CLE_GROQ_ICI", "tvly-votre_cle_tavily", True),  # placeholder long mais bien formé
        ("gsk_court", "tvly-court", False),
        ("sk-proj-autre-prefixe", "tvly-" + "x" * 30, True),
    ],
)
def test_check_keys_rejects_placeholders(groq: str, tavily: str, tavily_ok: bool) -> None:
    groq_item, tavily_item = core.check_keys(groq, tavily)
    assert groq_item.ok is False
    assert groq_item.hint, "une clé invalide doit proposer une correction"
    assert tavily_item.ok is tavily_ok


def test_check_corpus_reports_documents_and_sources() -> None:
    item = core.check_corpus()
    assert item.group == "Corpus"
    assert item.ok is True
    assert "4 documents" in item.detail
    for name in core.CORPUS:
        assert name in item.detail


def test_preflight_ready_ignores_non_critical_failures() -> None:
    blocking = core.CheckItem("Clés API", "GROQ_API_KEY", False, "absente", "hint", critical=True)
    optional = core.CheckItem("Clés API", "TAVILY_API_KEY", False, "absente", "", critical=False)

    assert core.preflight_ready([blocking, optional]) is False
    blocking.ok = True
    assert core.preflight_ready([blocking, optional]) is True


def test_preflight_groups_order_is_stable() -> None:
    items = core.preflight()
    groups = [item.group for item in items]
    assert groups[0] == "Configuration", "le .env doit être signalé en premier"
    assert "Paquets" in groups
    assert "Clés API" in groups
    assert groups[-1] in {"Corpus", "Index"}
    assert any(item.label.startswith("all-MiniLM") for item in items)


def test_env_source_never_leaks_the_key(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(core, "ENV_PATH", str(tmp_path / ".env"))
    source = core.env_source()
    assert ".env" in source
    assert "gsk_" not in source and "tvly-" not in source


def test_check_env_file_ok_when_loaded(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(core, "ENV_PATH", str(tmp_path / ".env"))
    monkeypatch.setattr(core, "ENV_EXAMPLE_FILE", str(tmp_path / "absent.example"))

    items = core.check_env_file()
    assert len(items) == 1
    assert items[0].ok is True
    assert items[0].group == "Configuration"


def test_check_env_file_reports_missing_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(core, "ENV_PATH", "")
    monkeypatch.setattr(core, "ENV_EXAMPLE_FILE", str(tmp_path / "absent.example"))

    item = core.check_env_file()[0]
    assert item.ok is False
    assert item.critical is True
    assert "introuvable" in item.detail
    assert ".env" in item.hint


def test_check_env_file_flags_real_keys_in_example(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    example = tmp_path / ".env.example"
    example.write_text(
        f"GROQ_API_KEY=gsk_{'a' * 40}\nTAVILY_API_KEY=tvly-votre_cle_tavily\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(core, "ENV_PATH", str(tmp_path / ".env"))
    monkeypatch.setattr(core, "ENV_EXAMPLE_FILE", str(example))

    items = core.check_env_file()
    flagged = [item for item in items if "clés réelles" in item.label]
    assert len(flagged) == 1
    assert flagged[0].ok is False
    assert flagged[0].critical is False  # avertissement, pas blocage
    assert "GROQ_API_KEY" in flagged[0].detail
    assert "TAVILY_API_KEY" not in flagged[0].detail  # le placeholder est ignoré


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("GROQ_API_KEY", "gsk_" + "a" * 40, True),
        ("GROQ_API_KEY", "gsk_votre_cle_groq", False),
        ("GROQ_API_KEY", "VOTRE_CLE_GROQ_ICI", False),
        ("GROQ_API_KEY", "", False),
        ("TAVILY_API_KEY", "tvly-" + "a" * 30, True),
        ("TAVILY_API_KEY", "tvly-votre_cle_tavily", False),
        ("AUTRE_CLE", "gsk_" + "a" * 40, False),
    ],
)
def test_looks_like_real_key(name: str, value: str, expected: bool) -> None:
    assert core._looks_like_real_key(name, value) is expected


# --- Export ------------------------------------------------------------------
def test_evaluation_json_is_serialisable() -> None:
    results = {item["id"]: core.RunResult(tools_used=item["expected_tools"]) for item in core.GOLDEN_SET}
    verdicts, score = core.routing_score(core.GOLDEN_SET, results)
    payload = json.loads(core.evaluation_json(verdicts, score, results))

    assert payload["score"] == 1.0
    assert payload["verdicts"]["Q5"] == "exact"
    assert payload["runs"]["Q1"]["tools_used"] == ["rag_search"]
    assert payload["runs"]["Q1"]["trace"] == []
