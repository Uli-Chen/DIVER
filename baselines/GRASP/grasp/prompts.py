"""Paper Fig. 5/14 templates, with separately documented filling conventions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return (ROOT/'prompts'/name).read_text().strip()


def render(controller, decision):
    c=controller.config
    if decision['kind']=='discovery':
        exclusions=sorted(controller.nodes)[-c['discovery_exclusion_cap']:]
        prompt=load('discovery.txt').replace('{TOPIC}',decision['frame']).replace('{KNOWN_ENTITIES}', '\n'.join(exclusions) or '(none)')
        # The original GRASP query is embedded in full. No hidden context is added.
        return {'query':prompt,'retrieval_query':prompt,'generation_query':prompt}
    target=decision['target']
    parts=[load('target.txt').replace('{target}',target)]
    if decision['template']!='baseline':
        memory=controller.active
        common=load('exclude.txt').replace('[EXCLUDE_RELATION_LIST]', '\n'.join(r.text() for r in sorted(memory.edges)) or '(none)')
        selector=load(decision['template']+'.txt')
        if decision['template']=='A':
            selector=selector.replace('{FRAME_HINTS}','{'+', '.join(decision['frame_hints'])+'}')
        selector=selector.replace('{OBS_TYPES}',', '.join(sorted(memory.types)) or '(none)')
        selector=selector.replace('{RESIDUAL_CAP}',str(c['residual_cap']))
        parts += [common,selector]
    parts.append(load('extraction.txt').replace('{target}',target))
    prompt='\n\n'.join(parts)
    return {'query':prompt,'retrieval_query':prompt,'generation_query':prompt}
