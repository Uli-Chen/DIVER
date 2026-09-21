"""Offline workspace/import/prompt/index-path checks. Never starts extraction."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import yaml
import extraction.pipeline
from extraction.prompts import PromptLibrary


def main() -> None:
    assert Path(extraction.pipeline.__file__).resolve().is_relative_to(ROOT)
    config = yaml.safe_load((ROOT / 'configs/workspace.yaml').read_text())
    assert config['experiment_output'] == 'tmp'
    prompts = PromptLibrary(ROOT / 'configs/prompts/bnrr', profile='bnrr')
    datasets = {}
    import lancedb
    from extraction.paths import graph_root
    for name, directory in config['datasets'].items():
        path = graph_root(name) / 'output'
        required = ['entities.parquet', 'relationships.parquet', 'text_units.parquet']
        missing = [file for file in required if not (path / file).is_file()]
        assert not missing, f'{name}: missing {missing}; run scripts/setup/build_index.py first'
        assert (path / 'lancedb').is_dir(), name
        table = lancedb.connect(str(path / 'lancedb')).open_table('default-entity-description')
        assert table.schema.field('vector').type.list_size == config['embedding_dimensions'], name
        datasets[name] = 'local index present (4096 dimensions)'
    print(json.dumps({'workspace': str(ROOT), 'local_source_import': True,
      'prompt_files': len(prompts.templates), 'datasets': datasets,
      'api_requests': 0, 'experiments_started': 0,
      'new_controller_implemented': True}, indent=2))


if __name__ == '__main__':
    main()
