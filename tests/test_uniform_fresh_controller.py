import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from extraction.config import ExperimentConfig
from extraction.control.controllers import (
    AgeaHubController,
    DegreeConditionedBnrrFreshController,
    EpochFewaController,
    FreshAnchorController,
    FrontierAlternatingFreshController,
    MinimumDegreeAlternatingFreshController,
    OpenEgoAlternatingFreshController,
)
from extraction.pipeline import MedicalExtractionPipeline, _arm_diagnostics


class FreshAnchorControllerTest(unittest.TestCase):
    def test_uses_every_positive_fresh_arm_before_repeating(self) -> None:
        controller = FreshAnchorController(seed=42)
        priors = {"ALPHA": 0.2, "BETA": 0.9, "GAMMA": 0.5}
        selected = []

        for turn in range(1, 4):
            arm = controller.select(priors, turn, priors)
            selected.append(arm)
            controller.observe(arm, 1.0)

        self.assertEqual(set(selected), set(priors))
        self.assertEqual(len(set(selected)), 3)
        self.assertEqual(
            [entry["reason"] for entry in controller.selection_history],
            ["uniform_fresh", "uniform_fresh", "uniform_fresh"],
        )

        repeated = controller.select(priors, 4, priors)
        self.assertIn(repeated, priors)
        self.assertEqual(
            controller.selection_history[-1]["reason"],
            "uniform_least_pulled",
        )

    def test_positive_bnrr_gate_and_arm_validation(self) -> None:
        controller = FreshAnchorController(seed=7)
        arms = ["ISOLATE", "SUMMARY", "CONNECTED"]
        priors = {"ISOLATE": 0.0, "SUMMARY": 0.8, "CONNECTED": 0.2}

        self.assertEqual(controller.select(arms, 1, priors), "CONNECTED")
        admission = controller.admission_history[-1]
        self.assertEqual(admission["eligible_arm_count"], 1)
        self.assertEqual(admission["eligibility_gate"], "bnrr_sensitivity>0")

    def test_seeded_selection_is_reproducible_and_reward_independent(self) -> None:
        priors = {f"ARM {index}": (index + 1) / 10 for index in range(10)}
        left = FreshAnchorController(seed=11)
        right = FreshAnchorController(seed=11)

        left_sequence = []
        right_sequence = []
        for turn in range(1, 11):
            left_arm = left.select(priors, turn, priors)
            right_arm = right.select(priors, turn, priors)
            left_sequence.append(left_arm)
            right_sequence.append(right_arm)
            left.observe(left_arm, 0.0)
            right.observe(right_arm, 1.0)

        self.assertEqual(left_sequence, right_sequence)
        self.assertFalse(left.state_dict()["reward_used_for_selection"])


class AgeaHubControllerTest(unittest.TestCase):
    @staticmethod
    def topology() -> dict[str, dict[str, int]]:
        return {
            "LEAF": {"degree": 1},
            "MID": {"degree": 10},
            "HUB": {"degree": 100},
            "ISOLATE": {"degree": 0},
        }

    def test_prefers_agea_hub_pool_and_obeys_pull_caps(self) -> None:
        topology = self.topology()
        controller = AgeaHubController(seed=4, candidate_k=1)
        first = controller.select(topology, 1, topology=topology)
        self.assertEqual(first, "HUB")
        controller.observe(first, 0.0)
        second = controller.select(topology, 2, topology=topology)
        self.assertEqual(second, "HUB")
        self.assertFalse(controller.state_dict()["reward_used_for_selection"])
        self.assertFalse(
            controller.state_dict()["hidden_query_generator_target_draw"]
        )

    def test_degree_below_twenty_is_eligible_only_once(self) -> None:
        topology = {"LOW": {"degree": 10}, "OTHER": {"degree": 2}}
        controller = AgeaHubController(seed=1, candidate_k=1)
        selected = controller.select(topology, 1, topology=topology)
        self.assertEqual(selected, "LOW")
        controller.observe(selected, 1.0)
        self.assertEqual(
            controller.select(topology, 2, topology=topology), "OTHER"
        )


class FrontierAlternatingFreshControllerTest(unittest.TestCase):
    def test_preserves_full_minimum_tie_group_and_global_freshness(self) -> None:
        controller = FrontierAlternatingFreshController(seed=42)
        priors = {
            **{f"FRONTIER {index}": 0.2 for index in range(6)},
            **{f"OTHER {index}": 0.3 + index / 10 for index in range(6)},
        }
        selected: list[str] = []
        scopes: list[str] = []

        for turn in range(1, 9):
            arm = controller.select(priors, turn, priors)
            self.assertIsNotNone(arm)
            selected.append(str(arm))
            record = controller.selection_history[-1]
            scopes.append(str(record["selection_scope"]))
            if record["frontier_forced"]:
                self.assertEqual(priors[str(arm)], record["minimum_sensitivity"])
            controller.observe(arm, float(turn % 2))

        self.assertEqual(len(selected), len(set(selected)))
        self.assertEqual(
            scopes,
            ["minimum_sensitivity_tie_group", "all"] * 4,
        )
        self.assertEqual(
            controller.admission_history[0]["frontier_pool_size"], 6
        )
        self.assertTrue(controller.admission_history[0]["tie_group_preserved"])
        self.assertFalse(controller.state_dict()["reward_used_for_selection"])

    def test_seeded_sequence_is_reward_independent(self) -> None:
        priors = {f"ARM {index}": 0.2 if index < 5 else 0.8 for index in range(10)}
        left = FrontierAlternatingFreshController(seed=11)
        right = FrontierAlternatingFreshController(seed=11)
        left_sequence: list[str | None] = []
        right_sequence: list[str | None] = []
        for turn in range(1, 9):
            left_arm = left.select(priors, turn, priors)
            right_arm = right.select(priors, turn, priors)
            left_sequence.append(left_arm)
            right_sequence.append(right_arm)
            left.observe(left_arm, 0.0)
            right.observe(right_arm, 1.0)
        self.assertEqual(left_sequence, right_sequence)


class MinimumDegreeAlternatingFreshControllerTest(unittest.TestCase):
    @staticmethod
    def topology() -> dict[str, dict[str, float | int]]:
        return {
            "LEAF A": {"degree": 1},
            "LEAF B": {"degree": 1},
            "MIDDLE": {"degree": 2},
            "HUB A": {"degree": 4},
            "HUB B": {"degree": 4},
            "ISOLATE": {"degree": 0},
        }

    def test_uses_degree_not_bnrr_prior_and_preserves_ties(self) -> None:
        controller = MinimumDegreeAlternatingFreshController(seed=42)
        topology = self.topology()
        # Deliberately reverse the degree ordering in the BNRR-like priors.
        priors = {
            "LEAF A": 0.99,
            "LEAF B": 0.98,
            "MIDDLE": 0.5,
            "HUB A": 0.01,
            "HUB B": 0.02,
            "ISOLATE": 1.0,
        }
        available = list(priors)

        first = controller.select(available, 1, priors, topology=topology)
        self.assertIn(first, {"LEAF A", "LEAF B"})
        admission = controller.admission_history[-1]
        selection = controller.selection_history[-1]
        self.assertEqual(admission["eligibility_gate"], "observed_degree>0")
        self.assertEqual(admission["eligible_arm_count"], 5)
        self.assertEqual(selection["selection_scope"], "minimum_degree_tie_group")
        self.assertEqual(selection["degree_pool_size"], 2)
        self.assertEqual(selection["minimum_degree"], 1)
        self.assertTrue(selection["tie_group_preserved"])
        self.assertNotIn("ISOLATE", [item["arm"] for item in admission["candidates"]])

        controller.observe(first, 100.0)
        second = controller.select(available, 2, priors, topology=topology)
        self.assertNotEqual(first, second)
        self.assertEqual(controller.selection_history[-1]["selection_scope"], "all")
        self.assertFalse(controller.state_dict()["reward_used_for_selection"])

    def test_seeded_sequence_is_reward_and_prior_independent(self) -> None:
        topology = self.topology()
        available = list(topology)
        left = MinimumDegreeAlternatingFreshController(seed=11)
        right = MinimumDegreeAlternatingFreshController(seed=11)
        left_sequence: list[str | None] = []
        right_sequence: list[str | None] = []
        for turn in range(1, 6):
            left_arm = left.select(
                available,
                turn,
                {arm: 0.01 for arm in available},
                topology=topology,
            )
            right_arm = right.select(
                available,
                turn,
                {arm: 0.99 for arm in available},
                topology=topology,
            )
            left_sequence.append(left_arm)
            right_sequence.append(right_arm)
            left.observe(left_arm, 0.0)
            right.observe(right_arm, 1.0)
        self.assertEqual(left_sequence, right_sequence)

    def test_requires_topology_for_every_available_arm(self) -> None:
        controller = MinimumDegreeAlternatingFreshController(seed=1)
        with self.assertRaisesRegex(ValueError, "Missing topology"):
            controller.select(["KNOWN", "MISSING"], 1, topology={"KNOWN": {"degree": 1}})


class DegreeConditionedBnrrFreshControllerTest(unittest.TestCase):
    @staticmethod
    def topology() -> dict[str, dict[str, float | int]]:
        return {
            "D1 A": {"degree": 1, "bnrr": 1.0},
            "D1 B": {"degree": 1, "bnrr": 1.0},
            "D3 LOW": {"degree": 3, "bnrr": 3**0.5},
            "D3 MID": {"degree": 3, "bnrr": (27 / 5) ** 0.5},
            "D3 HIGH": {"degree": 3, "bnrr": 3.0},
        }

    def test_treatment_uses_complete_minimum_bnrr_tie_within_degree(self) -> None:
        topology = self.topology()
        controller = DegreeConditionedBnrrFreshController(seed=4)

        selected = controller.select(topology, 1, topology=topology)

        record = controller.selection_history[-1]
        self.assertEqual(topology[str(selected)]["degree"], record["pivot_degree"])
        if record["pivot_degree"] == 3:
            self.assertEqual(selected, "D3 LOW")
            self.assertTrue(record["bnrr_informative"])
        else:
            self.assertIn(selected, {"D1 A", "D1 B"})
            self.assertFalse(record["bnrr_informative"])
            self.assertEqual(record["bnrr_policy_tv"], 0.0)
        self.assertTrue(record["degree_marginal_preserved"])
        self.assertTrue(record["tie_group_preserved"])

    def test_high_treatment_uses_complete_maximum_bnrr_tie_within_degree(self) -> None:
        topology = self.topology()
        controller = DegreeConditionedBnrrFreshController(
            policy_name="degree_conditioned_bnrr_high_fresh",
            variant="bnrr_high",
            seed=0,
        )

        selected = controller.select(topology, 1, topology=topology)

        record = controller.selection_history[-1]
        self.assertEqual(record["pivot_degree"], 3)
        self.assertEqual(selected, "D3 HIGH")
        self.assertEqual(selected, record["paired_bnrr_high_arm"])
        self.assertTrue(record["bnrr_informative"])
        self.assertTrue(record["degree_marginal_preserved"])
        self.assertEqual(record["maximum_bnrr_tie_group_size"], 1)

    def test_variants_use_common_random_numbers_and_pivot_degree(self) -> None:
        topology = self.topology()
        variants = [
            DegreeConditionedBnrrFreshController(seed=11),
            DegreeConditionedBnrrFreshController(
                policy_name="degree_conditioned_bnrr_high_fresh",
                variant="bnrr_high",
                seed=11,
            ),
            DegreeConditionedBnrrFreshController(
                policy_name="degree_conditioned_uniform_fresh",
                variant="uniform",
                seed=11,
            ),
            DegreeConditionedBnrrFreshController(
                policy_name="degree_conditioned_shuffled_bnrr_low_fresh",
                variant="shuffled_bnrr_low",
                seed=11,
            ),
            DegreeConditionedBnrrFreshController(
                policy_name="degree_conditioned_shuffled_bnrr_high_fresh",
                variant="shuffled_bnrr_high",
                seed=11,
            ),
        ]

        for controller in variants:
            controller.select(topology, 1, topology=topology)

        records = [controller.selection_history[-1] for controller in variants]
        self.assertEqual(len({record["pivot_u"] for record in records}), 1)
        self.assertEqual(len({record["selection_u"] for record in records}), 1)
        self.assertEqual(len({record["pivot_arm"] for record in records}), 1)
        self.assertEqual(len({record["pivot_degree"] for record in records}), 1)
        self.assertEqual(
            records[1]["selected_arm"], records[1]["paired_bnrr_high_arm"]
        )
        self.assertEqual(
            records[2]["selected_arm"], records[2]["paired_uniform_arm"]
        )
        self.assertEqual(
            records[3]["selected_arm"],
            records[3]["paired_shuffled_bnrr_low_arm"],
        )
        self.assertEqual(
            records[4]["selected_arm"],
            records[4]["paired_shuffled_bnrr_high_arm"],
        )

    def test_degree_one_naturally_reduces_to_identical_uniform_choice(self) -> None:
        topology = {
            "LEAF A": {"degree": 1, "bnrr": 1.0},
            "LEAF B": {"degree": 1, "bnrr": 1.0},
            "LEAF C": {"degree": 1, "bnrr": 1.0},
        }
        controller = DegreeConditionedBnrrFreshController(seed=7)

        controller.select(topology, 1, topology=topology)

        record = controller.selection_history[-1]
        self.assertEqual(
            record["paired_bnrr_low_arm"], record["paired_uniform_arm"]
        )
        self.assertEqual(
            record["paired_bnrr_low_arm"],
            record["paired_shuffled_bnrr_low_arm"],
        )
        self.assertEqual(
            record["paired_bnrr_low_arm"], record["paired_bnrr_high_arm"]
        )
        self.assertEqual(
            record["paired_bnrr_low_arm"],
            record["paired_shuffled_bnrr_high_arm"],
        )
        self.assertTrue(record["d1_stratum"])
        self.assertEqual(record["bnrr_policy_tv"], 0.0)

    def test_sequence_is_reward_value_independent(self) -> None:
        topology = self.topology()
        left = DegreeConditionedBnrrFreshController(seed=9)
        right = DegreeConditionedBnrrFreshController(seed=9)
        left_sequence: list[str | None] = []
        right_sequence: list[str | None] = []
        for turn in range(1, 6):
            left_arm = left.select(topology, turn, topology=topology)
            right_arm = right.select(topology, turn, topology=topology)
            left_sequence.append(left_arm)
            right_sequence.append(right_arm)
            left.observe(left_arm, 0.0)
            right.observe(right_arm, 100.0)
        self.assertEqual(left_sequence, right_sequence)
        self.assertFalse(left.state_dict()["reward_used_for_selection"])

    def test_shuffled_scores_preserve_within_degree_multiset(self) -> None:
        topology = self.topology()
        for variant in ("shuffled_bnrr_low", "shuffled_bnrr_high"):
            controller = DegreeConditionedBnrrFreshController(
                policy_name=f"degree_conditioned_{variant}_fresh",
                variant=variant,
                seed=4,
            )

            controller.select(topology, 1, topology=topology)

            stratum = [
                item
                for item in controller.admission_history[-1]["candidates"]
                if item["in_pivot_degree_stratum"]
            ]
            self.assertEqual(
                sorted(float(item["bnrr"]) for item in stratum),
                sorted(
                    float(item["shuffled_selection_score"]) for item in stratum
                ),
            )

    def test_invalid_or_missing_topology_fails_fast(self) -> None:
        controller = DegreeConditionedBnrrFreshController(seed=1)
        with self.assertRaisesRegex(ValueError, "Missing topology"):
            controller.select(
                ["KNOWN", "MISSING"],
                1,
                topology={"KNOWN": {"degree": 1, "bnrr": 1.0}},
            )
        with self.assertRaisesRegex(ValueError, "Invalid BNRR"):
            controller.select(
                ["BROKEN"],
                1,
                topology={"BROKEN": {"degree": 2, "bnrr": float("nan")}},
            )

    def test_summary_reports_activation_and_degree_matching(self) -> None:
        controller = DegreeConditionedBnrrFreshController(seed=4)
        topology = self.topology()
        for turn in range(1, 6):
            arm = controller.select(topology, turn, topology=topology)
            controller.observe(arm, 0.0)

        diagnostics = _arm_diagnostics(controller.state_dict())

        self.assertEqual(diagnostics["bnrr_selection_slots"], 5)
        self.assertGreaterEqual(diagnostics["bnrr_informative_slot_rate"], 0.0)
        self.assertGreaterEqual(diagnostics["mean_bnrr_policy_tv"], 0.0)
        self.assertEqual(diagnostics["degree_marginal_violation_count"], 0)
        self.assertAlmostEqual(
            diagnostics["fresh_slot_rate"]
            + diagnostics["least_pulled_slot_rate"],
            1.0,
        )
        self.assertEqual(
            sum(diagnostics["selected_degree_histogram"].values()), 5
        )


class ScheduledDegreeConditionedBnrrControllerTest(unittest.TestCase):
    @staticmethod
    def topology() -> dict[str, dict[str, float | int]]:
        return {
            "LOW": {"degree": 3, "bnrr": 1.5},
            "MID LOW": {"degree": 3, "bnrr": 2.0},
            "MID HIGH": {"degree": 3, "bnrr": 2.5},
            "HIGH A": {"degree": 3, "bnrr": 3.0},
            "HIGH B": {"degree": 3, "bnrr": 3.0},
        }

    def build(
        self,
        *,
        profile: str = "linear",
        variant: str = "bnrr_high_scheduled",
        seed: int = 5,
        schedule_start: float = 0.20,
        schedule_end: float = 0.70,
    ) -> DegreeConditionedBnrrFreshController:
        return DegreeConditionedBnrrFreshController(
            policy_name="degree_conditioned_bnrr_scheduled_fresh",
            variant=variant,
            schedule_profile=profile,
            schedule_total_turns=100,
            schedule_start_fraction=schedule_start,
            schedule_end_fraction=schedule_end,
            seed=seed,
        )

    def select_at(
        self, global_turn: int, *, profile: str = "linear"
    ) -> dict[str, object]:
        topology = self.topology()
        controller = self.build(profile=profile)
        controller.select(
            topology,
            1,
            topology=topology,
            global_turn=global_turn,
        )
        return controller.selection_history[-1]

    def test_linear_schedule_preserves_complete_threshold_ties(self) -> None:
        early = self.select_at(20)
        middle = self.select_at(45)
        late = self.select_at(70)

        self.assertEqual(early["schedule_phase"], "maximum")
        self.assertEqual(early["schedule_requested_tail_fraction"], 0.0)
        self.assertEqual(early["schedule_tail_pool_size"], 2)
        self.assertEqual(early["schedule_tail_cutoff"], 3.0)
        self.assertAlmostEqual(
            float(middle["schedule_requested_tail_fraction"]), 0.5
        )
        self.assertEqual(middle["schedule_tail_pool_size"], 3)
        self.assertEqual(middle["schedule_tail_cutoff"], 2.5)
        self.assertEqual(late["schedule_phase"], "uniform")
        self.assertEqual(late["schedule_requested_tail_fraction"], 1.0)
        self.assertEqual(late["schedule_tail_pool_size"], 5)
        self.assertTrue(early["tie_group_preserved"])

    def test_staged_schedule_has_frozen_phase_boundaries(self) -> None:
        expectations = {
            33: ("maximum", 0.0, 2),
            40: ("top_quartile", 0.25, 2),
            60: ("top_half", 0.50, 3),
            67: ("uniform", 1.0, 5),
        }
        for global_turn, expected in expectations.items():
            with self.subTest(global_turn=global_turn):
                record = self.select_at(global_turn, profile="staged")
                self.assertEqual(
                    (
                        record["schedule_phase"],
                        record["schedule_requested_tail_fraction"],
                        record["schedule_tail_pool_size"],
                    ),
                    expected,
                )

    def test_linear_schedule_accepts_pre_registered_compressed_window(self) -> None:
        topology = self.topology()
        expectations = {
            20: ("maximum", 0.0, 2),
            40: ("annealing", 0.5, 3),
            60: ("uniform", 1.0, 5),
        }
        for global_turn, expected in expectations.items():
            with self.subTest(global_turn=global_turn):
                controller = self.build(
                    schedule_start=0.20, schedule_end=0.60
                )
                controller.select(
                    topology,
                    1,
                    topology=topology,
                    global_turn=global_turn,
                )
                record = controller.selection_history[-1]
                self.assertEqual(record["schedule_phase"], expected[0])
                self.assertAlmostEqual(
                    float(record["schedule_requested_tail_fraction"]),
                    expected[1],
                )
                self.assertEqual(record["schedule_tail_pool_size"], expected[2])

    def test_raw_and_shuffled_schedules_match_degree_and_capacity(self) -> None:
        topology = self.topology()
        raw = self.build(seed=13)
        shuffled = self.build(
            variant="shuffled_bnrr_high_scheduled", seed=13
        )

        raw.select(topology, 1, topology=topology, global_turn=45)
        shuffled.select(topology, 1, topology=topology, global_turn=45)
        left = raw.selection_history[-1]
        right = shuffled.selection_history[-1]

        self.assertEqual(left["pivot_u"], right["pivot_u"])
        self.assertEqual(left["selection_u"], right["selection_u"])
        self.assertEqual(left["pivot_degree"], right["pivot_degree"])
        self.assertEqual(
            left["schedule_raw_tail_pool_size"],
            left["schedule_shuffled_tail_pool_size"],
        )
        self.assertEqual(
            right["schedule_raw_tail_pool_size"],
            right["schedule_shuffled_tail_pool_size"],
        )
        self.assertTrue(left["schedule_capacity_matched"])
        self.assertTrue(right["schedule_capacity_matched"])
        self.assertTrue(left["degree_marginal_preserved"])
        self.assertTrue(right["degree_marginal_preserved"])

    def test_scheduled_selection_is_reward_value_independent(self) -> None:
        topology = self.topology()
        left = self.build(seed=17)
        right = self.build(seed=17)
        left_sequence: list[str | None] = []
        right_sequence: list[str | None] = []
        for exploit_pull, global_turn in enumerate((10, 30, 50, 75), start=1):
            left_arm = left.select(
                topology,
                exploit_pull,
                topology=topology,
                global_turn=global_turn,
            )
            right_arm = right.select(
                topology,
                exploit_pull,
                topology=topology,
                global_turn=global_turn,
            )
            left_sequence.append(left_arm)
            right_sequence.append(right_arm)
            left.observe(left_arm, 0.0)
            right.observe(right_arm, 100.0)
        self.assertEqual(left_sequence, right_sequence)
        self.assertFalse(left.state_dict()["reward_used_for_selection"])

    def test_scheduled_variant_requires_global_turn(self) -> None:
        topology = self.topology()
        with self.assertRaisesRegex(ValueError, "require global_turn"):
            self.build().select(topology, 1, topology=topology)

    def test_hard_switch_moves_from_high_bnrr_to_degree_frontier(self) -> None:
        topology = {
            "LEAF A": {"degree": 1, "bnrr": 1.0},
            "LEAF B": {"degree": 1, "bnrr": 1.0},
            "D3 LOW": {"degree": 3, "bnrr": 1.5},
            "D3 HIGH": {"degree": 3, "bnrr": 3.0},
        }
        controller = DegreeConditionedBnrrFreshController(
            policy_name=(
                "degree_conditioned_bnrr_high_then_degree_frontier_fresh"
            ),
            variant="bnrr_high_then_dfa",
            schedule_profile="hard_switch",
            schedule_total_turns=100,
            schedule_start_fraction=0.0,
            schedule_end_fraction=0.20,
            seed=0,
        )

        early = controller.select(
            topology, 1, topology=topology, global_turn=20
        )
        early_record = controller.selection_history[-1]
        self.assertEqual(early_record["schedule_phase"], "maximum")
        self.assertEqual(early, early_record["paired_bnrr_high_arm"])
        self.assertTrue(early_record["degree_marginal_preserved"])
        controller.observe(early, 0.0)

        frontier = controller.select(
            topology, 2, topology=topology, global_turn=21
        )
        frontier_record = controller.selection_history[-1]
        self.assertEqual(
            frontier_record["schedule_phase"], "degree_frontier"
        )
        self.assertTrue(frontier_record["hybrid_degree_forced"])
        self.assertEqual(topology[str(frontier)]["degree"], 1)
        self.assertIn(frontier, frontier_record["active_arms"])
        self.assertTrue(frontier_record["degree_marginal_preserved"])
        self.assertTrue(frontier_record["schedule_capacity_matched"])
        controller.observe(frontier, 100.0)

        controller.select(topology, 3, topology=topology, global_turn=22)
        coverage_record = controller.selection_history[-1]
        self.assertEqual(coverage_record["schedule_phase"], "degree_frontier")
        self.assertFalse(coverage_record["hybrid_degree_forced"])
        self.assertEqual(
            coverage_record["hybrid_selection_pool_size"],
            coverage_record["base_pool_size"],
        )
        diagnostics = _arm_diagnostics(controller.state_dict())
        self.assertEqual(diagnostics["degree_marginal_violation_count"], 0)
        self.assertEqual(diagnostics["schedule_capacity_mismatch_count"], 0)
        self.assertEqual(
            diagnostics["schedule_phase_counts"],
            {"degree_frontier": 2, "maximum": 1},
        )

    def test_hybrid_switch_is_reward_value_independent(self) -> None:
        topology = {
            **self.topology(),
            "LEAF": {"degree": 1, "bnrr": 1.0},
        }
        controllers = [
            DegreeConditionedBnrrFreshController(
                policy_name=(
                    "degree_conditioned_bnrr_high_then_degree_frontier_fresh"
                ),
                variant="bnrr_high_then_dfa",
                schedule_profile="hard_switch",
                schedule_total_turns=100,
                schedule_start_fraction=0.0,
                schedule_end_fraction=0.20,
                seed=23,
            )
            for _ in range(2)
        ]
        sequences: list[list[str | None]] = [[], []]
        for exploit_pull, global_turn in enumerate((10, 21, 22, 23), start=1):
            for index, controller in enumerate(controllers):
                arm = controller.select(
                    topology,
                    exploit_pull,
                    topology=topology,
                    global_turn=global_turn,
                )
                sequences[index].append(arm)
                controller.observe(arm, float(index * 100))
        self.assertEqual(sequences[0], sequences[1])
        self.assertFalse(controllers[0].state_dict()["reward_used_for_selection"])

    def test_schedule_diagnostics_report_contract(self) -> None:
        topology = self.topology()
        controller = self.build()
        for exploit_pull, global_turn in enumerate((10, 30, 50, 75), start=1):
            arm = controller.select(
                topology,
                exploit_pull,
                topology=topology,
                global_turn=global_turn,
            )
            controller.observe(arm, 0.0)

        diagnostics = _arm_diagnostics(controller.state_dict())

        self.assertEqual(diagnostics["schedule_selection_slots"], 4)
        self.assertEqual(diagnostics["schedule_capacity_mismatch_count"], 0)
        self.assertEqual(diagnostics["degree_marginal_violation_count"], 0)
        self.assertGreater(
            diagnostics["mean_schedule_realized_tail_share"], 0.0
        )


class OpenEgoAlternatingFreshControllerTest(unittest.TestCase):
    @staticmethod
    def topology() -> dict[str, dict[str, float | int]]:
        return {
            "LEAF": {
                "degree": 1,
                "neighbor_edges": 0,
                "bnrr": 1.0,
                "openness": 1.0,
            },
            "OPEN TWO": {
                "degree": 2,
                "neighbor_edges": 0,
                "bnrr": 2.0,
                "openness": 1.0,
            },
            "CLOSED TWO": {
                "degree": 2,
                "neighbor_edges": 1,
                "bnrr": 2**0.5,
                "openness": 2**-0.5,
            },
            "OPEN THREE": {
                "degree": 3,
                "neighbor_edges": 0,
                "bnrr": 3.0,
                "openness": 1.0,
            },
        }

    def test_alternates_complete_open_tie_and_all_fresh(self) -> None:
        topology = self.topology()
        priors = {arm: 0.5 for arm in topology}
        controller = OpenEgoAlternatingFreshController(seed=42)
        selected: list[str] = []
        for turn in range(1, 5):
            arm = controller.select(priors, turn, priors, topology=topology)
            self.assertIsNotNone(arm)
            selected.append(str(arm))
            record = controller.selection_history[-1]
            if record["topology_forced"]:
                self.assertEqual(topology[str(arm)]["openness"], 1.0)
            controller.observe(arm, float(turn))

        self.assertEqual(len(selected), len(set(selected)))
        self.assertEqual(
            [record["selection_scope"] for record in controller.selection_history],
            ["maximum_openness_tie_group", "all"] * 2,
        )
        self.assertEqual(controller.admission_history[0]["open_pool_size"], 3)
        self.assertFalse(controller.state_dict()["reward_used_for_selection"])

    def test_supported_variant_excludes_leaves_only_from_topology_slot(self) -> None:
        topology = self.topology()
        priors = {arm: 0.5 for arm in topology}
        controller = OpenEgoAlternatingFreshController(
            seed=7,
            minimum_topology_degree=2,
            policy_name="supported_open_ego_alternating_fresh",
        )
        first = controller.select(priors, 1, priors, topology=topology)
        self.assertIn(first, {"OPEN TWO", "OPEN THREE"})
        controller.observe(first, 0.0)
        second = controller.select(priors, 2, priors, topology=topology)
        self.assertIn(second, set(priors) - {first})
        self.assertEqual(
            controller.state_dict()["minimum_topology_degree"], 2
        )

    def test_seeded_sequence_is_reward_independent(self) -> None:
        topology = self.topology()
        priors = {arm: 0.5 for arm in topology}
        left = OpenEgoAlternatingFreshController(seed=11)
        right = OpenEgoAlternatingFreshController(seed=11)
        left_sequence: list[str | None] = []
        right_sequence: list[str | None] = []
        for turn in range(1, 5):
            left_arm = left.select(priors, turn, priors, topology=topology)
            right_arm = right.select(priors, turn, priors, topology=topology)
            left_sequence.append(left_arm)
            right_sequence.append(right_arm)
            left.observe(left_arm, 0.0)
            right.observe(right_arm, 1.0)
        self.assertEqual(left_sequence, right_sequence)

    def test_structural_opportunity_uses_unqueried_neighbor_coverage(self) -> None:
        topology = self.topology()
        topology["LEAF"].update(
            pulled_neighbor_fraction=1.0,
            structural_opportunity=0.0,
        )
        topology["OPEN TWO"].update(
            pulled_neighbor_fraction=0.5,
            structural_opportunity=0.5,
        )
        topology["CLOSED TWO"].update(
            pulled_neighbor_fraction=0.0,
            structural_opportunity=2**-0.5,
        )
        topology["OPEN THREE"].update(
            pulled_neighbor_fraction=2 / 3,
            structural_opportunity=1 / 3,
        )
        priors = {arm: 0.5 for arm in topology}
        controller = OpenEgoAlternatingFreshController(
            seed=42,
            policy_name="structural_opportunity_fresh",
            topology_score_name="structural_opportunity",
            topology_slot_period=1,
        )

        selected = controller.select(priors, 1, priors, topology=topology)

        self.assertEqual(selected, "CLOSED TWO")
        record = controller.selection_history[-1]
        self.assertEqual(
            record["selection_scope"],
            "maximum_structural_opportunity_tie_group",
        )
        self.assertEqual(record["topology_pool_size"], 1)
        self.assertFalse(controller.state_dict()["reward_used_for_selection"])
        controller.observe(selected, 0.0)
        controller.select(priors, 2, priors, topology=topology)
        self.assertTrue(
            controller.selection_history[-1]["topology_forced"]
        )


class UniformFreshConfigurationTest(unittest.TestCase):
    def test_uniform_fresh_is_the_default_policy(self) -> None:
        self.assertEqual(
            ExperimentConfig().anchor_sampling_policy,
            "uniform_fresh",
        )

    def test_pipeline_routes_default_and_legacy_policies(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            def build(policy: str, run_id: str) -> MedicalExtractionPipeline:
                config = ExperimentConfig(
                    graph_root=str(root),
                    data_dir=str(root),
                    output_root=str(root / "runs"),
                    run_id=run_id,
                    anchor_sampling_policy=policy,
                )
                with (
                    patch("extraction.pipeline.TruthData.load", return_value=Mock()),
                    patch(
                        "extraction.pipeline.AgeaGraphRagAdapter",
                        return_value=Mock(),
                    ),
                ):
                    return MedicalExtractionPipeline(config)

            default_pipeline = build("uniform_fresh", "default")
            legacy_pipeline = build("ts_pl_fewa", "legacy")
            frontier_pipeline = build(
                "frontier_alternating_fresh", "frontier"
            )
            degree_pipeline = build(
                "minimum_degree_alternating_fresh", "minimum-degree"
            )
            agea_pipeline = build("agea_hub", "agea-hub")
            conditioned_pipeline = build(
                "degree_conditioned_bnrr_low_fresh", "conditioned-bnrr"
            )
            conditioned_high_pipeline = build(
                "degree_conditioned_bnrr_high_fresh", "conditioned-bnrr-high"
            )
            scheduled_pipeline = build(
                "degree_conditioned_bnrr_scheduled_fresh",
                "conditioned-bnrr-scheduled",
            )
            hybrid_pipeline = build(
                "degree_conditioned_bnrr_high_then_degree_frontier_fresh",
                "conditioned-bnrr-dfa-hybrid",
            )
            conditioned_uniform_pipeline = build(
                "degree_conditioned_uniform_fresh", "conditioned-uniform"
            )
            conditioned_shuffle_pipeline = build(
                "degree_conditioned_shuffled_bnrr_low_fresh",
                "conditioned-shuffle",
            )
            conditioned_shuffle_high_pipeline = build(
                "degree_conditioned_shuffled_bnrr_high_fresh",
                "conditioned-shuffle-high",
            )
            scheduled_shuffle_pipeline = build(
                "degree_conditioned_shuffled_bnrr_scheduled_fresh",
                "conditioned-scheduled-shuffle",
            )
            open_pipeline = build(
                "open_ego_alternating_fresh", "open"
            )
            supported_open_pipeline = build(
                "supported_open_ego_alternating_fresh", "supported-open"
            )
            structural_pipeline = build(
                "structural_opportunity_fresh", "structural"
            )

            self.assertIsInstance(
                default_pipeline.arm_controller,
                FreshAnchorController,
            )
            self.assertIsInstance(
                legacy_pipeline.arm_controller,
                EpochFewaController,
            )
            self.assertIsInstance(
                frontier_pipeline.arm_controller,
                FrontierAlternatingFreshController,
            )
            self.assertIsInstance(
                degree_pipeline.arm_controller,
                MinimumDegreeAlternatingFreshController,
            )
            self.assertIsInstance(agea_pipeline.arm_controller, AgeaHubController)
            self.assertIsInstance(
                conditioned_pipeline.arm_controller,
                DegreeConditionedBnrrFreshController,
            )
            self.assertEqual(conditioned_pipeline.arm_controller.variant, "bnrr_low")
            self.assertEqual(
                conditioned_high_pipeline.arm_controller.variant, "bnrr_high"
            )
            self.assertEqual(
                scheduled_pipeline.arm_controller.variant,
                "bnrr_high_scheduled",
            )
            self.assertEqual(
                scheduled_pipeline.arm_controller.schedule_total_turns,
                50,
            )
            self.assertIsInstance(
                hybrid_pipeline.arm_controller,
                DegreeConditionedBnrrFreshController,
            )
            self.assertEqual(
                hybrid_pipeline.arm_controller.variant,
                "bnrr_high_then_dfa",
            )
            self.assertEqual(
                conditioned_uniform_pipeline.arm_controller.variant, "uniform"
            )
            self.assertEqual(
                conditioned_shuffle_pipeline.arm_controller.variant,
                "shuffled_bnrr_low",
            )
            self.assertEqual(
                conditioned_shuffle_high_pipeline.arm_controller.variant,
                "shuffled_bnrr_high",
            )
            self.assertEqual(
                scheduled_shuffle_pipeline.arm_controller.variant,
                "shuffled_bnrr_high_scheduled",
            )
            self.assertIsInstance(
                open_pipeline.arm_controller,
                OpenEgoAlternatingFreshController,
            )
            self.assertEqual(
                open_pipeline.arm_controller.minimum_topology_degree,
                1,
            )
            self.assertIsInstance(
                supported_open_pipeline.arm_controller,
                OpenEgoAlternatingFreshController,
            )
            self.assertEqual(
                supported_open_pipeline.arm_controller.minimum_topology_degree,
                2,
            )
            self.assertIsInstance(
                structural_pipeline.arm_controller,
                OpenEgoAlternatingFreshController,
            )
            self.assertEqual(
                structural_pipeline.arm_controller.topology_score_name,
                "structural_opportunity",
            )
            self.assertEqual(
                structural_pipeline.arm_controller.topology_slot_period,
                1,
            )

    def test_pipeline_computes_structural_opportunity_from_pulled_neighbors(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = ExperimentConfig(
                graph_root=str(root),
                data_dir=str(root),
                output_root=str(root / "runs"),
                run_id="structural-features",
                anchor_sampling_policy=(
                    "structural_opportunity_fresh"
                ),
            )
            with (
                patch("extraction.pipeline.TruthData.load", return_value=Mock()),
                patch("extraction.pipeline.AgeaGraphRagAdapter", return_value=Mock()),
            ):
                pipeline = MedicalExtractionPipeline(config)
            pipeline.graph.add_edge("ALPHA", "BETA")
            pipeline.graph.add_edge("BETA", "GAMMA")
            pipeline.arm_controller.rewards["ALPHA"] = [0.0]

            _, _, topology = pipeline._available_arms()

            self.assertEqual(topology["BETA"]["pulled_neighbor_count"], 1)
            self.assertAlmostEqual(
                float(topology["BETA"]["pulled_neighbor_fraction"]), 0.5
            )
            self.assertAlmostEqual(
                float(topology["BETA"]["structural_opportunity"]), 0.5
            )


if __name__ == "__main__":
    unittest.main()
