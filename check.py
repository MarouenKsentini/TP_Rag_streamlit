"""Pré-vol en ligne de commande : paquets, clés API, corpus, cache d'embedding.

    uv run check.py

Les vérifications viennent de `agent_core.preflight()` — exactement celles
affichées dans le panneau « Diagnostic » de la barre latérale Streamlit.
Aucun import lourd, aucun appel réseau.
"""

from __future__ import annotations

import sys
from itertools import groupby

from agent_core import CheckItem, preflight, preflight_ready

RULE = "-" * 62


def render(items: list[CheckItem]) -> None:
    for group, group_items in groupby(items, key=lambda item: item.group):
        print(f"\n{group}")
        print(RULE)
        for item in group_items:
            print(f"  {item.icon} {item.label:<26} {item.detail}")
            if not item.ok and item.hint:
                print(f"     -> {item.hint}")


def main() -> int:
    print("TP 4.a - ReAct agent (Streamlit) - pre-vol")

    items = preflight()
    render(items)

    blocking = [item for item in items if item.critical and not item.ok]
    warnings = [item for item in items if not item.critical and not item.ok]

    print(f"\n{RULE}")
    print(f"  {len(items) - len(blocking) - len(warnings)}/{len(items)} vérifications OK")
    if warnings:
        print(f"  {len(warnings)} avertissement(s) non bloquant(s)")

    if blocking:
        print("\n  Corrige les points ❌ ci-dessus, puis relance ce script.")
        return 1

    print("\n  Prêt : uv run streamlit run app.py")
    print("         (le même diagnostic est visible dans la barre latérale de l'app)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
