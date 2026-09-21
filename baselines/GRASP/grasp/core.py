from __future__ import annotations
from collections import Counter, deque
from dataclasses import dataclass, field, asdict
import hashlib
import json
import random
import re


def rng(seed, *coordinates):
    value = json.dumps([seed, *coordinates], ensure_ascii=False).encode()
    return random.Random(int.from_bytes(hashlib.sha256(value).digest(), 'big'))


def entity(value):
    return ' '.join(value.strip().split()).upper()


@dataclass(frozen=True, order=True)
class Relation:
    source: str
    kind: str
    target: str

    def text(self):
        return f'<{self.source}> --[<{self.kind}>]--> <{self.target}>'


@dataclass
class Parsed:
    relations: list[Relation] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    status: str = 'parsed'
    rejected: int = 0
    explicit_none: bool = False


LINE = re.compile(r'^\s*-\s*\(([^()\r\n]+)\)\s*<([^<>\r\n]+)>\s*--\[<([^<>\r\n]+)>\]-->\s*<([^<>\r\n]+)>\s*$')
ENTITY_LINE = re.compile(r'^\s*-\s*<([^<>\r\n]+)>\s*$')
COMPAT_LINE = re.compile(r'^\s*-\s*\(([^()\r\n]+)\)\s*(.+?)\s*--\[(.+?)\]-->\s*(.+?)\s*$')
COMPAT_ENTITY_LINE = re.compile(r'^\s*-\s*(.+?)\s*$')


def unwrap(value):
    """Accept omitted presentation brackets, without repairing field contents."""
    value=value.strip()
    if value.startswith('<') and value.endswith('>'):value=value[1:-1].strip()
    return None if '<' in value or '>' in value else value


def parse_reply(text, target=None, *, discovery=False, truncated=False, verbatim=False):
    """Response-only parsing; never validates IDs/relations against retrieved truth."""
    if text.strip() == '[NONE]':
        return Parsed(status='truncated' if truncated else 'none', explicit_none=not truncated)
    start, end = ('[ENTITIES]', '[END ENTITIES]') if discovery else ('[RELATIONS]', '[END RELATIONS]')
    # Read each marked block independently, including repeated output blocks.
    # A truncated open block contributes newline-terminated records only:
    # the unfinished last line may contain a partially generated bare endpoint.
    pattern = re.escape(start) + r'(.*?)(' + re.escape(end) + r'|(?=' + re.escape(start) + r')|\Z)'
    bodies = []
    for match in re.finditer(pattern, text, re.DOTALL):
        body, closing = match.groups()
        if closing != end:
            if not truncated:
                continue
            body = body[:body.rfind('\n') + 1]
        bodies.append(body)
    if not bodies and not truncated and re.search(r'(?:^|\n)\s*\[NONE\]\s*\Z', text):
        # Accept explanatory prose followed by an explicit terminal marker,
        # but do not erase relation records merely because NONE also occurs.
        if not any(COMPAT_LINE.fullmatch(line) for line in text.splitlines()):
            return Parsed(status='none', explicit_none=True)
    if bodies and not truncated and all(body.strip()=='[NONE]' for body in bodies):
        return Parsed(status='none', explicit_none=True)
    if not bodies:
        return Parsed(status='truncated' if truncated else 'unparseable')
    result = Parsed()
    identity = (lambda value:value.strip()) if verbatim else entity
    for line in '\n'.join(bodies).splitlines():
        if not line.strip() or line.strip()=='[NONE]': continue
        match = (COMPAT_ENTITY_LINE if discovery else COMPAT_LINE).fullmatch(line)
        if not match:
            result.rejected += 1
            continue
        if discovery:
            raw=unwrap(match[1])
            if raw is None:
                result.rejected += 1
                continue
            label = identity(raw)
            if not label or label.casefold() in {'null', 'none', 'unknown'}:
                result.rejected += 1
            elif label not in result.entities:
                result.entities.append(label)
            continue
        ident, source, kind, dest = match.groups()
        source,kind,dest=map(unwrap,(source,kind,dest))
        if any(x is None for x in [source,kind,dest]):
            result.rejected += 1
            continue
        atom = Relation(identity(source), kind.strip() if verbatim else ' '.join(kind.split()), identity(dest))
        if any(not x or x.casefold() in {'null','none','unknown'} for x in [atom.source,atom.kind,atom.target]):
            result.rejected += 1
            continue
        if target is not None and identity(target) not in {atom.source, atom.target}:
            result.rejected += 1
            continue
        # IDs do not veto otherwise valid records. Keep raw duplicate records
        # for Good-Turing; the controller still deduplicates relation identity.
        result.relations.append(atom)
    if not discovery:
        result.entities = sorted({v for r in result.relations for v in [r.source,r.target]})
    if truncated:
        result.status = 'partial' if result.entities else 'truncated'
    elif result.rejected:
        result.status = 'partial' if result.entities else 'unparseable'
    return result


def regime(value):
    return 'surge' if value > 2 else 'stall' if value < .5 else 'steady'


def policy(state):
    """Table VII, in printed priority order; normalize only after reweighting."""
    scar, sat = len(state.types) < 3, len(state.types) >= 6
    mode = regime(state.ema)
    if state.none_streak == 1: return 'none1', (.7,0,0,.3)
    if state.none_streak >= 2: return 'none2', (.3,0,0,.7)
    if mode == 'stall':
        if state.zero_streak >= 3:
            return ('stall3_scar',(.5,0,.2,.3)) if scar else ('stall3_rich',(.5,.2,0,.3))
        # The printed rich row sums to .9; preserve it, then normalize.
        return ('stall_scar',(.3,0,.5,.2)) if scar else ('stall_rich',(.3,.3,0,.3))
    if mode == 'surge':
        if scar: return 'surge_scar', (0,0,.5,.5)
        if state.last_types == 0 or sat: return 'surge_exploit', (0,1,0,0)
        return 'surge', (0,.5,0,.5)
    if scar: return 'steady_scar', (0,0,1,0)
    if state.last_types == 0 and state.last_edges > 0: return 'steady_exploit', (0,.7,0,.3)
    if sat: return 'steady_sat', (0,1,0,0)
    return 'steady', (.05,.35,.35,.25)


@dataclass
class Scheduler:
    target: str
    config: dict
    count: int = 0
    edges: set[Relation] = field(default_factory=set)
    types: set[str] = field(default_factory=set)
    ema: float = 0
    template_ema: dict = field(default_factory=lambda: {x:.5 for x in 'ABCD'})
    diversity: bool = False
    none_streak: int = 0
    zero_streak: int = 0
    last_edges: int = 0
    last_types: int = 0
    window: list[list[Relation]] = field(default_factory=list)

    def novelty(self):
        if not self.window: return None
        frequencies = Counter(r for batch in self.window for r in batch)
        return sum(n == 1 for n in frequencies.values()) / len(self.window)

    def stop_reason(self):
        if self.count >= self.config['per_target_budget']: return 'target_budget'
        if self.count >= self.config['warmup'] and self.novelty() < self.config['stop_threshold']:
            return 'good_turing'
        return None

    def choose(self, seed, global_turn):
        if self.stop_reason(): raise RuntimeError('Target episode already stopped')
        estimate = self.novelty()
        if estimate is not None and estimate < self.config['diversity_threshold']:
            self.diversity = True
        if not self.diversity:
            return {'template':'baseline','policy':'baseline','weights':{},'novelty':estimate,'regime':regime(self.ema)}
        name, base = policy(self)
        weights = {k:w*(.25+1.5*self.template_ema[k]+.5*abs(self.template_ema[k]-.5)) for k,w in zip('ABCD',base)}
        total = sum(weights.values())
        weights = {k:v/total for k,v in weights.items()}
        sample = rng(seed,'template',self.target,global_turn).random()
        cumulative, selected = 0.0, 'D'
        for k,w in weights.items():
            cumulative += w
            if sample < cumulative:
                selected = k
                break
        return {'template':selected,'policy':name,'weights':weights,'novelty':estimate,'regime':regime(self.ema)}

    def observe(self, parsed, template):
        old_regime = regime(self.ema)
        unique = set(parsed.relations)
        new = unique-self.edges
        new_types = {r.kind for r in new}-self.types
        self.edges |= unique
        self.types |= {r.kind for r in unique}
        self.last_edges, self.last_types = len(new),len(new_types)
        self.none_streak = self.none_streak+1 if parsed.explicit_none else 0
        self.zero_streak = self.zero_streak+1 if not new else 0
        alpha = self.config['ema_alpha']
        self.ema = alpha*len(new)+(1-alpha)*self.ema
        if template in self.template_ema:
            self.template_ema[template] = alpha*bool(new)+(1-alpha)*self.template_ema[template]
        if regime(self.ema) != old_regime:
            self.template_ema = {k:.5*v+.25 for k,v in self.template_ema.items()}
        self.window.append(list(parsed.relations))
        self.window = self.window[-self.config['novelty_window']:]
        self.count += 1


class Controller:
    """Native target episodes plus an explicitly non-paper FIFO frontier wrapper."""
    def __init__(self, config, frame_selector=None):
        self.config = config
        self.turn = 0
        self.nodes, self.edges = set(), set()
        self.queue, self.queued, self.finished = deque(), set(), set()
        self.active = None
        self.discovery_count = 0
        self.transitions = []
        self.pending = None
        self.frame_selector = frame_selector
        self.enqueue(config.get('targets',[]))

    def enqueue(self, labels):
        identity=(lambda value:value.strip()) if self.config['parser'].get('entity_identity')=='verbatim' else entity
        if self.config['frontier']=='fixed_target_cohort':
            labels=list(dict.fromkeys(identity(x) for x in labels))
            labels=[x for x in labels if x not in self.queued and x not in self.finished]
        else:
            labels = sorted({identity(x) for x in labels}-{*self.queued,*self.finished})
        if self.active: labels=[x for x in labels if x != self.active.target]
        if self.config['mode'] == 'targeted' and self.turn:
            return
        if self.config['frontier']!='fixed_target_cohort':
            rng(self.config['seed'],'frontier',self.turn).shuffle(labels)
        for label in labels:
            self.queue.append(label);self.queued.add(label)

    def next_request(self, render):
        if self.pending is not None: raise RuntimeError('Observe the pending request first')
        if self.turn >= self.config['global_budget']: return None
        if self.active and self.active.stop_reason():
            self.transitions.append({'turn':self.turn,'target':self.active.target,'reason':self.active.stop_reason(),'queries':self.active.count})
            self.finished.add(self.active.target);self.active=None
        if self.active is None and self.queue:
            label=self.queue.popleft();self.queued.remove(label)
            self.active=Scheduler(label,self.config['scheduler'])
        if self.active is None:
            if self.config['mode']=='targeted': return None
            frames=self.config['frames']
            decision={'kind':'discovery','target':None,'template':'discovery','frame':frames[self.discovery_count%len(frames)],'episode_round':None}
            self.discovery_count+=1
        else:
            decision={'kind':'target','target':self.active.target,'episode_round':self.active.count+1,
                **self.active.choose(self.config['seed'],self.turn+1)}
            if decision['template']=='A':
                if self.frame_selector is None:
                    from .frames import selector_from_config
                    self.frame_selector=selector_from_config(self.config)
                selection=self.frame_selector.select(self.config['frames'],self.active.types)
                decision['frame_hints']=selection['frames']
                decision['frame_selection']=selection
        decision['turn']=self.turn+1
        request={**decision,**render(self,decision)}
        self.pending=request
        return request

    def observe(self, reply):
        if self.pending is None: raise RuntimeError('No pending request')
        request=self.pending
        if reply.get('error'):
            parsed=Parsed(status='request_error')
        else:
            parsed=parse_reply(reply.get('response',''),request['target'],discovery=request['kind']=='discovery',
                truncated='length' in reply.get('finish_reasons',[]),
                verbatim=self.config['parser'].get('entity_identity')=='verbatim')
        if self.active:self.active.observe(parsed,request['template'])
        self.nodes.update(parsed.entities);self.edges.update(parsed.relations)
        self.turn+=1
        self.enqueue(parsed.entities)
        self.pending=None
        return {'turn':self.turn,'status':parsed.status,'rejected_records':parsed.rejected,
            'parsed_relations':[asdict(r) for r in parsed.relations],
            'observed_nodes':len(self.nodes),'observed_typed_edges':len(self.edges),
            'observed_directed_pairs':len({(r.source,r.target) for r in self.edges}),
            'scheduler':None if self.active is None else {'target':self.active.target,'queries':self.active.count,
                'ema':self.active.ema,'novelty':self.active.novelty(),'template_ema':self.active.template_ema.copy(),
                'stop_reason':self.active.stop_reason()}}

    def export(self):
        return {'rounds':self.turn,'nodes':sorted(self.nodes),'relations':[asdict(r) for r in sorted(self.edges)],
            'finished_targets':sorted(self.finished),'discovery_rounds':self.discovery_count,'transitions':self.transitions}
