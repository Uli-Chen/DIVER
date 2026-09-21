# Legacy compatibility prompts

`triaction/` preserves the 13 previous templates, byte-for-byte, for the inherited
tri-action pipeline and its regression tests. They are not loaded by the
BNRR-centric pilot. The active seven-file bundle is `configs/prompts/bnrr/`.

AGEA's native prompts remain in `baselines/AGEA/agea_prompts.py`; changing or
deleting them would alter the reference method. Target GraphRAG index/system
prompts are also not part of this cleanup.
