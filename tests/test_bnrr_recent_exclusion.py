import json
from pathlib import Path

import networkx as nx

from extraction.bnrr_prompt_profiles import load_profile
from extraction.bnrr_queries import exploration_messages, exploitation_messages
from extraction.query_recent_exclusion import recent_exclusion_memory

ROOT = Path(__file__).resolve().parents[1]


def test_common_means_repeated_responses_not_degree_or_mentions():
    rows = [{"query_intent": f"Q{i}", "returned_entity_names": ["A", "a", "A", f"ONCE{i}"]}
            for i in range(15)]
    result = recent_exclusion_memory(rows)
    assert result == {"recent_questions": ["Q12", "Q13", "Q14"], "avoid_as_main_topic": ["A"]}
    rows += [{"query_intent": "new", "returned_entity_names": []} for _ in range(10)]
    assert recent_exclusion_memory(rows)["avoid_as_main_topic"] == []


def test_missing_response_history_cannot_invent_frequency():
    rows = [{"query": "Q\n\nDO_NOT_COPY", "newly_discovered_entity_names": ["A"]}] * 5
    assert recent_exclusion_memory(rows) == {"recent_questions": ["Q"] * 3, "avoid_as_main_topic": []}
    assert recent_exclusion_memory([]) == {"recent_questions": [], "avoid_as_main_topic": []}


def test_exclusion_cap_and_ties_are_deterministic():
    rows = [{"returned_entity_names": list("ABCDEFGH")} for _ in range(2)]
    assert recent_exclusion_memory(rows)["avoid_as_main_topic"] == list("ABCDE")

