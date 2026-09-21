"""PIDE benchmark adapter with seed-stable static anchors."""
from baselines._shared.common import StaticAttack
from baselines.PIDE.prompt import PIDE_prompt_formatter

def prompt(anchor):
    return PIDE_prompt_formatter(anchor).strip()

class Attack(StaticAttack):
    def __init__(self, seed, anchors):
        super().__init__('PIDE', seed, anchors, prompt)
