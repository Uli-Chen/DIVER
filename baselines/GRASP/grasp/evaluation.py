"""Offline target-cohort evaluation (GRASP §III-B/VI-A); no attack access."""
import random
from collections import defaultdict
from pathlib import Path


def typed_truth(output, type_column):
    import pandas as pd
    df = pd.read_parquet(Path(output)/'relationships.parquet')
    if not type_column or type_column not in df.columns:
        raise ValueError('RType evaluation requires an explicit, configured relation-type column')
    triples=set()
    for s,r,t in df[['source',type_column,'target']].itertuples(index=False,name=None):
        if not all(isinstance(v,str) and v.strip() for v in (s,r,t)):
            raise ValueError('Typed truth contains a missing entity/type field')
        triples.add((s,r,t))
    return triples


def sample_targets(triples, seed, count=50, min_degree=5):
    # Directed in+out degree over unique endpoint pairs; eligibility is an
    # evaluator-owned cohort selection, not hidden information for the attacker.
    degree=defaultdict(int)
    for s,t in {(s,t) for s,_,t in triples}:
        degree[s]+=1;degree[t]+=1
    eligible=sorted(n for n,d in degree.items() if d>=min_degree)
    if len(eligible)<count:raise ValueError('Insufficient degree-eligible target entities')
    return random.Random(seed).sample(eligible,count)


def prf(predicted,truth):
    matched=len(predicted & truth)
    p=matched/len(predicted) if predicted else 0.0
    r=matched/len(truth) if truth else 0.0
    return {'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0.0,
            'predicted':len(predicted),'truth':len(truth),'matched':matched}


def evaluate_targets(triples, targets, predictions):
    per_target={}
    for target in targets:
        truth={e for e in triples if target in (e[0],e[2])}
        if not truth:raise ValueError('Target has no truth relations')
        pred=set(predictions.get(target,set()))
        # Preserve wrong predictions in denominators. Inputs retain the parser's
        # identities; this paper-profile evaluator performs no aliases/fuzzy match.
        per_target[target]={'RType':prf(pred,truth),
            'Naive':prf({(s,t) for s,_,t in pred},{(s,t) for s,_,t in truth})}
    if not per_target:raise ValueError('Empty target cohort')
    macro={kind:{metric:sum(row[kind][metric] for row in per_target.values())/len(per_target)
                 for metric in ['precision','recall','f1']} for kind in ['RType','Naive']}
    return {'scope':'per_target_macro','targets':per_target,'macro':macro}


def evaluate_run(config, out):
    from .io import read
    predictions=defaultdict(set)
    for path in sorted((Path(out)/'receipts').glob('turn_*.json')):
        receipt=read(path);target=receipt['request']['target']
        if target is not None:
            predictions[target].update((r['source'],r['kind'],r['target']) for r in receipt['result']['parsed_relations'])
    triples=typed_truth(Path(out)/'graph_root/output',config['evaluation']['type_column'])
    return evaluate_targets(triples,config['targets'],predictions)
