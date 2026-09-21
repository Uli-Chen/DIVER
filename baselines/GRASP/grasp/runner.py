from pathlib import Path
import json, os, time
from .core import Controller
from .prompts import render
from .io import inside, read, write, sha


def run(config,out,victim,resume=False):
    out=inside(out);receipts=out/'receipts';receipts.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (out/'.worker.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('A worker already holds this run') from None
        existing=sorted(receipts.glob('turn_*.json'))
        if existing and not resume:raise FileExistsError('Started run requires explicit resume')
        if [p.name for p in existing]!=[f'turn_{i:04}.json' for i in range(1,len(existing)+1)]:raise RuntimeError('Noncontiguous receipt history')
        # An interrupted request may have been billed. No automatic requery of an unresolved turn.
        for marker in receipts.glob('inflight_*.json'):
            receipt=marker.with_name(marker.name.replace('inflight_','turn_'))
            if not receipt.exists():raise RuntimeError('Unresolved request outcome: inspect inflight marker and HTTP ledger before resuming')
        controller=Controller(config)
        initial_count=len(existing)
        while True:
            checkpoint=out/'PAUSE_AFTER_ROUND.json'
            paused=False
            while victim is not None and checkpoint.exists() and controller.turn>=read(checkpoint)['round']:
                if not paused:
                    write(out/'STATUS.json',{'status':'paused_for_output_audit','round':controller.turn,'pid':os.getpid(),'updated':time.time()})
                    write(out/'checkpoint_recovered.json',controller.export())
                    print(f'GRASP paused after round {controller.turn} for output audit',flush=True)
                    paused=True
                time.sleep(1)
            if paused:write(out/'STATUS.json',{'status':'running','round':controller.turn,'pid':os.getpid(),'updated':time.time()})
            request=controller.next_request(render)
            if request is None:break
            path=receipts/f'turn_{request["turn"]:04}.json'
            if path.exists():
                old=read(path)
                if old['request']!=request:raise RuntimeError('Replay query/decision mismatch')
                response=old['reply']
                result=controller.observe(response)
                if old['result']!=result:raise RuntimeError('Replay result/state mismatch')
            else:
                if victim is None:break
                marker=receipts/f'inflight_{request["turn"]:04}.json'
                write(marker,{'turn':request['turn'],'request':request,'started':time.time()})
                response=victim.query(request)
                result=controller.observe(response)
                write(path,{'request':request,'reply':response,'result':result})
                write(out/'COMMIT.json',{'turn':request['turn'],'receipt_sha256':sha(path)})
                print(f"GRASP turn={request['turn']} kind={request['kind']} template={request['template']} nodes={result['observed_nodes']} pairs={result['observed_directed_pairs']} status={result['status']}",flush=True)
            if victim:write(out/'STATUS.json',{'status':'running','round':controller.turn,'pid':os.getpid(),'updated':time.time()})
        if controller.turn<initial_count:raise RuntimeError('Receipt history exceeds controller budget')
        recovered=controller.export()
        if victim:write(out/'recovered.json',recovered)
        write(out/('STATUS.json' if victim else 'REPLAY.json'),{'status':'completed' if victim else 'replayed','round':controller.turn,'budget':config['global_budget'],
            'stop_reason':'global_budget' if controller.turn==config['global_budget'] else 'targets_exhausted' if victim else 'replay_prefix',
            'updated':time.time()})
        return recovered


def offline_evaluate(config,out):
    """Ground truth is imported only here, after attack decisions are complete."""
    if config['evaluation']['primary']=='per_target_macro':
        from .evaluation import evaluate_run
        result=evaluate_run(config,out)
        write(Path(out)/'EVALUATION.json',result)
        return result
    import sys
    sys.path.insert(0,str(Path(config['source_code'])/'src'))
    import networkx as nx
    from evaluation.graph_recovery import TruthData,evaluate_recovery
    truth=TruthData.load(Path(out)/'graph_root/output')
    recovered=read(Path(out)/'recovered.json')
    graph=nx.MultiDiGraph();graph.add_nodes_from(recovered['nodes'])
    for r in recovered['relations']:graph.add_edge(r['source'],r['target'],relation=r['kind'])
    result={'scope':'global_untyped_directed_pairs','metrics':evaluate_recovery(graph,truth),'rounds':recovered['rounds'],
        'typed_fidelity':'not_measured; existing comparator does not supply a typed-edge matching protocol'}
    write(Path(out)/'EVALUATION.json',result)
    return result
