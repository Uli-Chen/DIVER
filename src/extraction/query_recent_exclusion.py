"""Minimal response-visible memory; no entity examples, descriptions or scores."""
from collections import Counter

from .models import normalize_label

RECENT_QUESTIONS = 3
ENTITY_WINDOW = 10
MAX_EXCLUSIONS = 5
MIN_OBSERVED_ROUNDS = 2


def recent_questions(history):
    return [str(h.get("query_intent") or h.get("query", "").split("\n\n", 1)[0])[:320]
            for h in history[-RECENT_QUESTIONS:]]


def recent_exclusion_memory(history):
    counts = Counter()
    for h in history[-ENTITY_WINDOW:]:
        # First-discovery lists cannot estimate recurrence. Missing response
        # observations contribute no exclusions, never invented frequencies.
        labels = {normalize_label(v) for v in h.get("returned_entity_names", [])}
        counts.update(v for v in labels if v)
    common = [v for v in sorted(counts, key=lambda v: (-counts[v], v))
              if counts[v] >= MIN_OBSERVED_ROUNDS][:MAX_EXCLUSIONS]
    return {"recent_questions": recent_questions(history), "avoid_as_main_topic": common}
