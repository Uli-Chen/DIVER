import unittest
from pathlib import Path
from unittest.mock import Mock

import networkx as nx

from extraction.control.actions import ActionKind, EntityPair, QueryAction
from extraction.control.triaction import LearningFreeTriActionController
from extraction.models import CandidateBatch
from extraction.pipeline import _action_batch_metrics
from extraction.prompts import PromptLibrary


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROMPT_DIR = PROJECT_DIR / "baselines" / "_shared" / "legacy_prompts" / "triaction"


def closure_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_edge("ALPHA", "BETA", key="ab", rel="ab")
    graph.add_edge("ALPHA", "GAMMA", key="ac", rel="ac")
    return graph


class TriActionCandidateTest(unittest.TestCase):
    def test_open_pairs_are_unordered_and_exclude_known_either_direction(self) -> None:
        controller = LearningFreeTriActionController(seed=3)
        graph = closure_graph()
        snapshot = controller.build_snapshot(graph)

        self.assertIn("ALPHA", snapshot.closure_executable)
        self.assertIn(
            EntityPair("BETA", "GAMMA"), snapshot.open_pairs["ALPHA"]
        )
        self.assertEqual(
            EntityPair("GAMMA", "BETA"), EntityPair("BETA", "GAMMA")
        )

        graph.add_edge("GAMMA", "BETA", key="gb", rel="gb")
        closed = controller.build_snapshot(graph)
        self.assertNotIn("ALPHA", closed.open_pairs)

    def test_bnrr_gate_is_conditioned_on_executable_degree_bins(self) -> None:
        controller = LearningFreeTriActionController(
            variant="bnrr_closure", seed=5, closure_bnrr_percentile=0.5
        )
        snapshot = controller.build_snapshot(closure_graph())

        self.assertTrue(snapshot.closure_eligible)
        self.assertTrue(
            set(snapshot.closure_eligible).issubset(snapshot.closure_executable)
        )
        for node in snapshot.closure_eligible:
            self.assertGreaterEqual(
                snapshot.degree_conditioned_percentiles[node], 0.5
            )
            self.assertGreaterEqual(snapshot.degree_bin_sizes[node], 1)

    def test_singleton_executable_bin_has_midrank_half(self) -> None:
        graph = nx.MultiDiGraph()
        graph.add_edge("CENTER", "LEFT", key="l")
        graph.add_edge("CENTER", "RIGHT", key="r")
        controller = LearningFreeTriActionController(seed=7)
        snapshot = controller.build_snapshot(graph)

        # CENTER is the only degree-two ego in its log2 degree bin.
        self.assertEqual(snapshot.degree_bin_sizes["CENTER"], 1)
        self.assertEqual(snapshot.degree_conditioned_percentiles["CENTER"], 0.5)
        self.assertIn("CENTER", snapshot.closure_eligible)

    def test_no_consecutive_closure_while_incident_is_available(self) -> None:
        controller = LearningFreeTriActionController(seed=11)
        graph = closure_graph()

        first = controller.select_local(controller.build_snapshot(graph), turn=2)
        second = controller.select_local(controller.build_snapshot(graph), turn=3)
        third = controller.select_local(controller.build_snapshot(graph), turn=4)

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertIsNotNone(third)
        self.assertEqual(first.kind, ActionKind.INCIDENT)
        self.assertEqual(second.kind, ActionKind.CLOSURE)
        self.assertEqual(third.kind, ActionKind.INCIDENT)
        self.assertEqual(sum(controller.incident_pull_counts.values()), 2)
        self.assertEqual(sum(controller.closure_anchor_pull_counts.values()), 1)

    def test_incident_remains_available_after_fresh_pool_is_exhausted(self) -> None:
        controller = LearningFreeTriActionController(
            variant="incident_only", seed=13
        )
        graph = closure_graph()
        snapshot = controller.build_snapshot(graph)
        controller.incident_pull_counts.update(
            {node: 1 for node in snapshot.incident_eligible}
        )

        exhausted = controller.build_snapshot(graph)
        action = controller.select_local(exhausted, turn=9)

        self.assertFalse(exhausted.fresh_incident)
        self.assertIsNotNone(action)
        self.assertEqual(action.kind, ActionKind.INCIDENT)
        self.assertEqual(
            action.metadata["reason"], "uniform_least_pulled_incident"
        )

    def test_text_unit_overlap_admits_and_ranks_pairs_without_truth(self) -> None:
        graph = nx.MultiDiGraph()
        for neighbor in ("A", "B", "C"):
            graph.add_edge("CENTER", neighbor)
        controller = LearningFreeTriActionController(
            variant="bnrr_closure",
            seed=17,
            closure_pair_policy="text_unit_overlap",
            closure_min_shared_text_units=1,
            max_closure_local_share=1.0,
            entity_text_units={
                "A": frozenset({"u1", "u2"}),
                "B": frozenset({"u1", "u2"}),
                "C": frozenset({"u1"}),
            },
        )

        snapshot = controller.build_snapshot(graph)
        action = controller.select_local(snapshot, turn=2)

        self.assertIsNotNone(action)
        self.assertEqual(action.kind, ActionKind.CLOSURE)
        self.assertEqual(action.pairs, (EntityPair("A", "B"),))
        self.assertEqual(action.metadata["pair_evidence_scores"], [2])
        self.assertEqual(action.metadata["closure_pair_policy"], "text_unit_overlap")

    def test_quarter_share_uses_one_closure_per_four_local_gaps(self) -> None:
        graph = nx.MultiDiGraph()
        for neighbor in ("A", "B", "C", "D", "E"):
            graph.add_edge("CENTER", neighbor)
        controller = LearningFreeTriActionController(
            variant="uniform_closure",
            seed=19,
            max_closure_local_share=0.25,
        )

        kinds = []
        for turn in range(2, 7):
            action = controller.select_local(
                controller.build_snapshot(graph), turn=turn
            )
            self.assertIsNotNone(action)
            kinds.append(action.kind)

        self.assertEqual(
            kinds,
            [
                ActionKind.INCIDENT,
                ActionKind.INCIDENT,
                ActionKind.INCIDENT,
                ActionKind.CLOSURE,
                ActionKind.INCIDENT,
            ],
        )

    def test_capacity_bundle_requires_a_full_admitted_bundle(self) -> None:
        graph = nx.MultiDiGraph()
        for neighbor in ("A", "B", "C"):
            graph.add_edge("CENTER", neighbor)
        controller = LearningFreeTriActionController(
            variant="bnrr_closure",
            closure_bundle_size=4,
            closure_pair_policy="text_unit_overlap",
            closure_min_shared_text_units=1,
            entity_text_units={
                "A": frozenset({"u"}),
                "B": frozenset({"u"}),
                "C": frozenset({"u"}),
            },
        )

        snapshot = controller.build_snapshot(graph)

        self.assertEqual(snapshot.raw_open_pair_counts["CENTER"], 3)
        self.assertNotIn("CENTER", snapshot.open_pairs)
        self.assertNotIn("CENTER", snapshot.closure_executable)


class PromptLibraryTest(unittest.TestCase):
    def test_all_action_prompts_load_from_one_bundle(self) -> None:
        library = PromptLibrary(PROMPT_DIR)
        common = {
            "dataset": "medical",
            "turn": 2,
            "seed_query": "seed",
            "topic": "treatments",
            "recent_queries": "- seed",
            "anchor": "CENTER",
            "degree": 2,
            "known_outgoing": "- CENTER -> LEFT [related_to]",
            "known_incoming": "- RIGHT -> CENTER [related_to]",
            "numbered_pairs": "1. LEFT -> RIGHT",
            "entity_a": "LEFT",
            "entity_b": "RIGHT",
            "neighbor_context": "- CENTER -> LEFT [related_to]",
        }
        actions = (
            QueryAction(ActionKind.GLOBAL, topic="treatments"),
            QueryAction(ActionKind.INCIDENT, anchor="CENTER"),
            QueryAction(
                ActionKind.CLOSURE,
                anchor="CENTER",
                pairs=(EntityPair("LEFT", "RIGHT"),),
            ),
        )

        for action in actions:
            rendered = library.render(action, values=common)
            self.assertIn(action.kind.value.upper(), rendered.text)
            if action.kind == ActionKind.CLOSURE:
                self.assertIn("EGO CLOSURE OUTPUT CONTRACT", rendered.text)
                self.assertNotIn("EVIDENCE AND OUTPUT CONTRACT", rendered.text)
                self.assertIn("[[NO_SUPPORTED_RELATIONSHIP]]", rendered.text)
                self.assertNotIn("CENTER", rendered.text)
            else:
                self.assertIn("EVIDENCE AND OUTPUT CONTRACT", rendered.text)
            self.assertTrue(Path(rendered.template_path).is_file())

    def test_seed_prompt_comes_from_seed_file(self) -> None:
        library = PromptLibrary(PROMPT_DIR)
        action = QueryAction(
            ActionKind.GLOBAL,
            topic="seed",
            metadata={"seed": True},
        )
        rendered = library.render(
            action,
            seed=True,
            values={
                "dataset": "medical",
                "turn": 1,
                "seed_query": "discover diseases",
            },
        )
        self.assertEqual(rendered.template_name, "seed")
        self.assertIn("ACTION: SEED_DISCOVERY", rendered.text)

    def test_bundle_closure_uses_separate_minimal_prompt_and_contract(self) -> None:
        library = PromptLibrary(PROMPT_DIR)
        action = QueryAction(
            ActionKind.CLOSURE,
            anchor="CENTER",
            pairs=(EntityPair("A", "B"), EntityPair("C", "D")),
        )
        rendered = library.render(
            action,
            values={
                "dataset": "medical",
                "turn": 3,
                "pair_questions": '1. "A" and "B"\n2. "C" and "D"',
            },
        )

        self.assertEqual(rendered.template_name, "closure_bundle")
        self.assertIn("ACTION: EGO_CLOSURE_BUNDLE", rendered.text)
        self.assertIn('1. "A" and "B"', rendered.text)
        self.assertIn("NO_SUPPORTED_RELATIONSHIP", rendered.text)
        self.assertNotIn("CENTER", rendered.text)
        self.assertTrue(rendered.template_path.endswith("ego_closure_bundle.txt"))

    def test_grounded_closure_uses_packet_json_prompt_and_contract(self) -> None:
        library = PromptLibrary(PROMPT_DIR)
        action = QueryAction(
            ActionKind.CLOSURE,
            anchor="CENTER",
            pairs=(EntityPair("A", "B"),),
            metadata={"grounded_evidence": True},
        )
        rendered = library.render(
            action,
            values={
                "dataset": "medical",
                "turn": 4,
                "anchor": "CENTER",
                "evidence_packets": '[{"left":"A","right":"B"}]',
            },
        )

        self.assertEqual(rendered.template_name, "closure_evidence")
        self.assertIn("EVIDENCE_PACKETS", rendered.text)
        self.assertIn('"supported": true', rendered.text)
        self.assertIn('[{"left":"A","right":"B"}]', rendered.text)
        self.assertTrue(
            rendered.output_contract_path.endswith(
                "ego_closure_evidence_output_contract.txt"
            )
        )

    def test_grounded_prompt_variants_share_the_strict_contract(self) -> None:
        library = PromptLibrary(PROMPT_DIR)
        expected_templates = {
            "coarse": "closure_evidence_coarse",
            "agea_exploit": "closure_evidence_agea",
            "fine_entailment": "closure_evidence_entailment",
        }
        for variant, expected_template in expected_templates.items():
            action = QueryAction(
                ActionKind.CLOSURE,
                anchor="CENTER",
                pairs=(EntityPair("A", "B"),),
                metadata={
                    "grounded_evidence": True,
                    "grounded_prompt_variant": variant,
                },
            )
            rendered = library.render(
                action,
                values={
                    "dataset": "medical",
                    "turn": 4,
                    "anchor": "CENTER",
                    "evidence_packets": '[{"left":"A","right":"B"}]',
                },
            )
            self.assertEqual(rendered.template_name, expected_template)
            self.assertIn("EGO CLOSURE GROUNDED JSON OUTPUT CONTRACT", rendered.text)
            self.assertTrue(
                rendered.output_contract_path.endswith(
                    "ego_closure_evidence_output_contract.txt"
                )
            )

    def test_unknown_grounded_prompt_variant_fails_closed(self) -> None:
        library = PromptLibrary(PROMPT_DIR)
        action = QueryAction(
            ActionKind.CLOSURE,
            anchor="CENTER",
            pairs=(EntityPair("A", "B"),),
            metadata={
                "grounded_evidence": True,
                "grounded_prompt_variant": "unknown",
            },
        )
        with self.assertRaisesRegex(ValueError, "Unknown grounded closure"):
            library.render(
                action,
                values={
                    "dataset": "medical",
                    "turn": 4,
                    "anchor": "CENTER",
                    "evidence_packets": "[]",
                },
            )


class UnorderedPairMetricTest(unittest.TestCase):
    def test_multiple_relation_types_count_as_one_primary_pair(self) -> None:
        batch = CandidateBatch.from_records(
            [],
            [
                {"source": "LEFT", "target": "RIGHT", "rel": "treats"},
                {"source": "LEFT", "target": "RIGHT", "rel": "supports"},
            ],
        )
        truth = Mock()
        truth.nodes = {"LEFT", "RIGHT"}
        truth.edges = {("LEFT", "RIGHT")}
        truth.canonical.side_effect = lambda value: value
        action = QueryAction(
            ActionKind.CLOSURE,
            anchor="CENTER",
            pairs=(EntityPair("LEFT", "RIGHT"),),
        )

        metrics = _action_batch_metrics(
            graph=nx.MultiDiGraph(),
            batch=batch,
            action=action,
            truth=truth,
        )

        self.assertEqual(metrics["validated_directed_pairs"], 1)
        self.assertEqual(metrics["new_directed_pairs"], 1)
        self.assertEqual(metrics["credited_new_directed_pairs"], 1)

    def test_reverse_direction_is_credited_for_unordered_closure_pair(self) -> None:
        batch = CandidateBatch.from_records(
            [], [{"source": "RIGHT", "target": "LEFT", "rel": "supports"}]
        )
        truth = Mock()
        truth.nodes = {"LEFT", "RIGHT"}
        truth.edges = {("RIGHT", "LEFT")}
        truth.canonical.side_effect = lambda value: value
        action = QueryAction(
            ActionKind.CLOSURE,
            anchor="CENTER",
            pairs=(EntityPair("LEFT", "RIGHT"),),
        )

        metrics = _action_batch_metrics(
            graph=nx.MultiDiGraph(), batch=batch, action=action, truth=truth
        )

        self.assertEqual(metrics["credited_new_directed_pairs"], 1)
        self.assertEqual(
            metrics["credited_pair_list"], [["RIGHT", "LEFT"]]
        )


if __name__ == "__main__":
    unittest.main()
