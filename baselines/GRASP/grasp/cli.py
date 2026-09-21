import argparse, json, os, shlex, shutil, subprocess, sys, time, traceback
from pathlib import Path
from .io import ROOT, inside, read, write, sha
from .config import check_config, inspect, prepare, validate, require_approval
from .boundary import install, kernel

RMUX=os.environ.get('RMUX', shutil.which('rmux') or '/opt/homebrew/bin/rmux')
SOCKET=ROOT/'.rmux'


def rmux(*args):return [RMUX,'-S',str(SOCKET),*args]


def command(action,out,approval,session,resume=False):
    profile=ROOT/'runtime/online.sb'
    profile.write_text('(version 1) (allow default) (deny file-write*) (allow file-write* (subpath '+json.dumps(str(ROOT))+'))')
    args=['/usr/bin/sandbox-exec','-f',str(profile),sys.executable,'-B','-u',str(ROOT/'grasp.py'),action,
        '--run',str(out),'--approval',str(approval),'--session',session,'--execute-online']
    if resume:args.append('--resume')
    return args


def launch(a):
    c,_=validate(a.run)
    subprocess.run([RMUX,'-V'],check=True,capture_output=True,text=True)
    session=a.session
    if not session:session=f'grasp-{c["dataset"]}-seed{c["seed"]}-{time.strftime("%Y%m%d-%H%M%S")}'
    import re
    if not re.fullmatch('[A-Za-z0-9_-]{1,90}',session):raise ValueError('Invalid session name')
    if (a.run/'LAUNCH.json').exists():raise FileExistsError('Use a separate new cohort for a relaunch')
    args=command('supervise',a.run,a.approval,session,a.resume)
    record={'session':session,'command':args,'created':time.time(),'attach':shlex.join(rmux('attach','-t',session))}
    write(a.run/'LAUNCH.json',record)
    try:subprocess.run(rmux('new-session','-d','-s',session,'exec '+shlex.join(args)+' 2>>'+shlex.quote(str(a.run/'supervisor.stderr.log'))),check=True)
    except Exception:
        write(a.run/'TERMINAL.json',{'status':'launch_failed','diagnostics':'No fallback multiplexer used','time':time.time()})
        raise
    return record


def supervise(a):
    validate(a.run)
    if read(a.run/'LAUNCH.json')['session']!=a.session:raise ValueError('Unowned session')
    start=time.time()
    with (a.run/'worker.log').open('ab',buffering=0) as log:
        p=subprocess.Popen(command('worker',a.run,a.approval,a.session,a.resume),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        write(a.run/'PROCESS.json',{'supervisor_pid':os.getpid(),'worker_pid':p.pid,'session':a.session,'started':start})
        while p.poll() is None:
            status=read(a.run/'STATUS.json') if (a.run/'STATUS.json').exists() else {}
            print(f"GRASP {status.get('status','starting')} round={status.get('round',0)} worker={p.pid}",flush=True)
            time.sleep(10)
        code=p.wait()
    status=read(a.run/'STATUS.json')
    write(a.run/'TERMINAL.json',{'status':'completed' if code==0 and status.get('status')=='completed' else 'failed',
        'exit_code':code,'rounds':status.get('round',0),'started':start,'finished':time.time(),'session':a.session,'log':str(a.run/'worker.log')})
    subprocess.run(rmux('kill-session','-t',a.session),check=False)


def worker(a):
    c,_=validate(a.run)
    from .runner import run
    from .transport import NativeVictim
    victim=NativeVictim(c,a.run)
    try:result=run(c,a.run,victim,a.resume)
    finally:victim.close()
    from extraction.request_audit import cost_summary
    requests=[json.loads(x) for x in (a.run/'requests.jsonl').read_text().splitlines()]
    write(a.run/'COSTS.json',cost_summary(requests))
    from .runner import offline_evaluate
    offline_evaluate(c,a.run)
    return {'rounds':result['rounds'],'nodes':len(result['nodes']),'typed_relations':len(result['relations'])}


def main():
    p=argparse.ArgumentParser(description='GRASP: offline by default; user audit required before every approved run.')
    p.add_argument('action',choices=['sample-targets','inspect','prepare','validate','replay','evaluate','launch','supervise','worker'])
    p.add_argument('--config',type=Path,default=ROOT/'configs/paper_targeted.example.json')
    p.add_argument('--output-config',type=Path,default=ROOT/'configs/paper_targeted.selected.json')
    p.add_argument('--run',type=Path,default=ROOT/'runs/grasp_run')
    p.add_argument('--approval',type=Path)
    p.add_argument('--session')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--execute-online',action='store_true')
    a=p.parse_args();a.run=inside(a.run);a.config=a.config.absolute()
    if a.approval is not None:a.approval=inside(a.approval)
    online=a.action in ['launch','supervise','worker']
    if online:
        if not a.execute_online:raise PermissionError('No experiment authorized: --execute-online omitted')
        require_approval(a.run,a.approval)
    install(online)
    if a.action!='launch':kernel(online)
    if a.action=='sample-targets':
        from .evaluation import typed_truth, sample_targets
        c=read(a.config)
        if c['profile']!='paper_targeted':raise ValueError('Target sampling requires paper profile')
        triples=typed_truth(Path(c['source_graph'])/'output',c['evaluation']['type_column'])
        selection=c['target_selection']
        c['targets']=sample_targets(triples,c['seed'],selection['count'],selection['min_degree'])
        check_config(c)
        out=inside(a.output_config)
        if out.exists():raise FileExistsError('Selected target configuration already exists')
        write(out,c);result={'config':str(out),'targets':len(c['targets']),'model_calls':0}
    elif a.action=='inspect':result=inspect(read(a.config))
    elif a.action=='prepare':result=prepare(a.config,a.run)
    elif a.action=='validate':validate(a.run);result={'status':'valid','model_calls':0}
    elif a.action=='replay':
        from .runner import run
        c,_=validate(a.run);result=run(c,a.run,None,True)
    elif a.action=='evaluate':
        from .runner import offline_evaluate
        c,_=validate(a.run);result=offline_evaluate(c,a.run)
    else:result=globals()[a.action](a)
    if result is not None:print(json.dumps(result,ensure_ascii=False,indent=2))
