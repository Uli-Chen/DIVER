"""Shared source loader and static-query state for GraphRAG baselines."""
import ast
import random

def definitions(path, namespace, names=None):
    tree = ast.parse(path.read_text())
    definitions = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                   and (names is None or n.name in names)]
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *definitions], type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)

class StaticAttack:
    def __init__(self, method, seed, anchors, prompt):
        self.method = method
        self.prompt = prompt
        self.anchors = list(anchors)
        random.Random(seed).shuffle(self.anchors)
        self.position = 0

    def next_query(self):
        if self.position >= len(self.anchors):
            raise RuntimeError('Static anchor pool exhausted')
        anchor = self.anchors[self.position]
        self.position += 1
        return {'anchor': anchor, 'query': self.prompt(anchor),
                'retrieval_query': anchor, 'is_mutation': False}

    def feedback(self, query, answer, turn, final=False):
        return {}

    def state(self):
        return {'position': self.position}

    def restore(self, state):
        self.position = state['position']
