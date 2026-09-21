"""Recent/exclusion memory with legacy-sized observed semantic descriptions."""
from .query_recent_exclusion import recent_exclusion_memory, recent_questions


def exploration_context(graph, history):
    context = recent_exclusion_memory(history)
    labels = []
    for h in history[-2:]:
        for name in h.get('newly_discovered_entity_names', []):
            if name in graph and name not in labels:
                labels.append(name)
    if not labels:
        labels = list(graph.nodes)[-10:]
    context['observed_entities'] = [{'name': name, 'description': str(graph.nodes[name].get('description', '')).
        split('\nRelationships:', 1)[0][:240]} for name in labels[:10]]
    return context


def exploitation_context(graph, history, anchor):
    connections, seen = [], set()
    for source, target, attrs in graph.edges(data=True):
        if anchor not in (source, target) or (source, target) in seen:
            continue
        seen.add((source, target))
        connections.append({'source': source, 'target': target, 'description': str(attrs.get('description', ''))[:160]})
        if len(connections) == 20:
            break
    return {'fixed_target': anchor, 'recent_questions': recent_questions(history),
        'observed_description': str(graph.nodes[anchor].get('description', '')).split('\nRelationships:', 1)[0][:480],
        'partial_observed_connections': connections}
