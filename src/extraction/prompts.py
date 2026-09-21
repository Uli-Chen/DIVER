"""File-backed prompt bundle for extraction experiment actions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .control.actions import ActionKind, QueryAction


PROMPT_FILES = {
    "seed": "seed.txt",
    ActionKind.GLOBAL.value: "global_discovery.txt",
    ActionKind.INCIDENT.value: "incident_expansion.txt",
    ActionKind.CLOSURE.value: "ego_closure.txt",
    "closure_bundle": "ego_closure_bundle.txt",
    "closure_evidence": "ego_closure_evidence.txt",
    "closure_evidence_coarse": "ego_closure_evidence_coarse.txt",
    "closure_evidence_agea": "ego_closure_evidence_agea.txt",
    "closure_evidence_entailment": "ego_closure_evidence_entailment.txt",
    "output_contract": "output_contract.txt",
    "closure_output_contract": "ego_closure_output_contract.txt",
    "closure_bundle_output_contract": "ego_closure_bundle_output_contract.txt",
    "closure_evidence_output_contract": (
        "ego_closure_evidence_output_contract.txt"
    ),
}

BNRR_PROMPT_FILES = {
    "seed": "seed.txt",
    ActionKind.GLOBAL.value: "global_discovery.txt",
    ActionKind.INCIDENT.value: "incident_expansion.txt",
    "output_contract": "output_contract.txt",
    "explore_query_generator": "explore_query_generator.txt",
    "exploit_query_generator": "exploit_query_generator.txt",
    "query_generator_system": "query_generator_system.txt",
}


class _StrictFormatValues(dict[str, Any]):
    def __missing__(self, key: str) -> Any:
        raise KeyError(f"Prompt template requires missing value {key!r}")


@dataclass(frozen=True)
class RenderedPrompt:
    text: str
    template_name: str
    template_path: str
    output_contract_path: str


class PromptLibrary:
    """Load all action prompts once and render them with strict placeholders."""

    def __init__(self, prompt_dir: str | Path, *, profile: str = "legacy") -> None:
        self.root = Path(prompt_dir).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Prompt directory does not exist: {self.root}")
        if profile not in {"legacy", "bnrr"}:
            raise ValueError(f"Unknown prompt profile: {profile}")
        self.profile = profile
        files = BNRR_PROMPT_FILES if profile == "bnrr" else PROMPT_FILES
        self.paths = {name: self.root / filename for name, filename in files.items()}
        missing = [str(path) for path in self.paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Prompt bundle is incomplete: {missing}")
        self.templates = {
            name: path.read_text(encoding="utf-8").strip()
            for name, path in self.paths.items()
        }
        if any(not template for template in self.templates.values()):
            empty = [name for name, value in self.templates.items() if not value]
            raise ValueError(f"Prompt bundle contains empty templates: {empty}")

    def render(
        self,
        action: QueryAction,
        *,
        values: Mapping[str, Any],
        seed: bool = False,
    ) -> RenderedPrompt:
        if self.profile == "bnrr" and not seed and action.kind == ActionKind.CLOSURE:
            raise ValueError("BNRR-centric has no closure action")
        if seed:
            template_name = "seed"
        elif (
            action.kind == ActionKind.CLOSURE
            and action.metadata.get("grounded_evidence")
        ):
            variant = str(
                action.metadata.get("grounded_prompt_variant", "baseline")
            )
            variant_templates = {
                "baseline": "closure_evidence",
                "coarse": "closure_evidence_coarse",
                "agea_exploit": "closure_evidence_agea",
                "fine_entailment": "closure_evidence_entailment",
            }
            try:
                template_name = variant_templates[variant]
            except KeyError as error:
                raise ValueError(
                    f"Unknown grounded closure prompt variant: {variant!r}"
                ) from error
        elif action.kind == ActionKind.CLOSURE and len(action.pairs) > 1:
            template_name = "closure_bundle"
        else:
            template_name = action.kind.value
        template = self.templates[template_name]
        payload = _StrictFormatValues(values)
        domain_prompt = template.format_map(payload).strip()
        if (
            action.kind == ActionKind.CLOSURE
            and not seed
            and action.metadata.get("grounded_evidence")
        ):
            contract_name = "closure_evidence_output_contract"
        elif action.kind == ActionKind.CLOSURE and not seed:
            contract_name = (
                "closure_bundle_output_contract"
                if len(action.pairs) > 1
                else "closure_output_contract"
            )
        else:
            contract_name = "output_contract"
        output_contract = self.templates[contract_name].format_map(payload).strip()
        return RenderedPrompt(
            text=f"{domain_prompt}\n\n{output_contract}",
            template_name=template_name,
            template_path=str(self.paths[template_name]),
            output_contract_path=str(self.paths[contract_name]),
        )
