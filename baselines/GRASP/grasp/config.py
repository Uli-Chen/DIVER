from pathlib import Path
import copy, json, shutil, sys
from .io import ROOT, PROJECT_ROOT, inside, read, write, sha, tree_hash, implementation_hashes


def encoder_path(c):
    p=Path(c['frame_encoder']['model_path'])
    return p if p.is_absolute() else PROJECT_ROOT/p


def check_config(c):
    # Resolve published relative paths against the repository, not the CLI cwd.
    for key in ['source_graph', 'source_index', 'source_code']:
        if key in c:
            p = Path(c[key]).expanduser()
            c[key] = str(p if p.is_absolute() else PROJECT_ROOT / p)
    if c['protocol']!='grasp-v2-semantic-frames-v2':raise ValueError('Historical GRASP configurations require explicit migration')
    if c['profile'] not in ['paper_targeted','benchmark_whole_graph']:raise ValueError('Unknown profile')
    if c['mode'] not in ['whole_graph','targeted']:raise ValueError('Unknown mode')
    for key in ['global_budget','residual_cap','discovery_exclusion_cap']:
        if type(c[key]) is not int or c[key]<1:raise ValueError('Invalid '+key)
    if type(c['seed']) is not int:raise ValueError('Seed must be an integer')
    if not c['frames'] or any(not isinstance(x,str) or not x.strip() for x in c['frames']):raise ValueError('Empty frame list')
    if c['mode']=='targeted' and not c['targets']:raise ValueError('Targeted mode requires user-provided targets')
    if c['mode']=='whole_graph' and c['targets']:raise ValueError('Whole-graph initialization must discover targets')
    s=c['scheduler']
    for key in ['per_target_budget','warmup','novelty_window']:
        if type(s[key]) is not int or s[key]<1:raise ValueError('Invalid scheduler integer')
    if s['novelty_window']<s['warmup']:raise ValueError('Window must accommodate warmup')
    if not 0<s['ema_alpha']<=1 or not 0<=s['stop_threshold']<s['diversity_threshold']:raise ValueError('Invalid scheduler bounds')
    if c['query_routing']!='full_grasp_prompt_to_retrieval_and_generation':raise ValueError('Query routing not implemented')
    if c['frontier'] not in ['fifo_seeded_batch_shuffle_no_revisit','fixed_target_cohort']:raise ValueError('Unknown frontier')
    if c['frame_hint_strategy']!='three_dissimilar_semantic_frames':raise ValueError('Semantic frame selection required')
    f=c['frame_encoder']
    if not f['model_path'] or f['aggregation']!='mean_pairwise_cosine_distance' or f['empty_history']!='configured_order' or f['tie_break']!='configured_order' or f['hint_rendering']!='whole_frame_phrases':raise ValueError('Unsupported encoder convention')
    if len(set(c['frames']))<3 or len(set(c['frames']))!=len(c['frames']):raise ValueError('At least three distinct, nonduplicate frames required')
    if c['profile']=='paper_targeted':
        if c['mode']!='targeted' or c['frontier']!='fixed_target_cohort':raise ValueError('Paper profile requires a fixed target cohort')
        if c['evaluation']['primary']!='per_target_macro' or not c['evaluation']['type_column']:raise ValueError('Paper profile requires typed evaluation')
        if c['parser'].get('entity_identity')!='verbatim':raise ValueError('Paper profile requires verbatim identity')
        if len(set(c['targets']))!=len(c['targets']):raise ValueError('Duplicate target cohort entries')
        if c['global_budget']<len(c['targets'])*s['per_target_budget']:raise ValueError('Global cap must not truncate the fixed target cohort')
        if c['victim']['output_tokens']!=2048 or c['victim']['extra_safeguard']!='paper_safe_prompt':raise ValueError('Paper profile requires its output budget and safe prompt')
    if not c['execution']['approval_required']:raise ValueError('User approval is mandatory')
    v=c['victim']
    if v['sdk_max_retries']!=2 or v['outer_retries'] or v['thinking']:raise ValueError('Expected comparator SDK retry setting 2, no outer retries, thinking disabled')
    if v['temperature']!=0 or v['top_p']!=1:raise ValueError('Expected comparator temperature=0 and top_p=1')
    return c


def inspect(c):
    check_config(c)
    if not encoder_path(c).is_dir():raise FileNotFoundError('Local frame encoder is not provisioned')
    import yaml
    import pyarrow.parquet as pq
    graph=Path(c['source_graph']);index=Path(c['source_index'])
    original=yaml.safe_load((graph/'settings.yaml').read_text())
    from graphrag.config.models.local_search_config import LocalSearchConfig
    effective=LocalSearchConfig(**original['local_search']).model_dump()
    v=c['victim']
    for key,expected in [('top_k_entities',v['top_k_entities']),('top_k_relationships',v['top_k_relationships']),('max_context_tokens',v['context_tokens'])]:
        if effective[key]!=expected:raise ValueError('Comparator retrieval mismatch: '+key)
    if c['profile']=='benchmark_whole_graph' and original['models']['default_chat_model']['max_tokens']!=v['output_tokens']:raise ValueError('Comparator output limit mismatch')
    for file in ['entities.parquet','relationships.parquet']:
        if sha(graph/'output'/file)!=sha(index/file):raise ValueError('Index differs from selected comparator truth: '+file)
    if not (index/'lancedb').is_dir():raise ValueError('Vector index missing')
    columns=pq.read_schema(index/'relationships.parquet').names
    if c['profile']=='paper_targeted':
        from .evaluation import typed_truth
        typed_truth(index,c['evaluation']['type_column'])
    return {'dataset':c['dataset'],'entity_rows':pq.read_metadata(index/'entities.parquet').num_rows,
        'relationship_rows':pq.read_metadata(index/'relationships.parquet').num_rows,
        'relationship_columns':columns,'explicit_type_column':any(x in columns for x in ['relation_type','rel_type','type']),
        'graph_matches_comparator':True,'retrieval_matches_comparator':True,
        'paper_type_schedule_caveat':'Types are taken only from emitted strings; no extra graph type field is created.',
        'model_calls':0}


def prepare(config_path, out):
    c=check_config(read(config_path));out=inside(out)
    if out.exists():raise FileExistsError('Output already exists; use a new run directory')
    info=inspect(c)
    import yaml
    graph=out/'graph_root';graph.mkdir(parents=True)
    index=Path(c['source_index']);source=Path(c['source_graph'])
    shutil.copytree(index,graph/'output')
    shutil.copytree(source/'prompts',graph/'prompts')
    settings=yaml.safe_load((source/'settings.yaml').read_text())
    if c['profile']=='paper_targeted':
        prompt_path=Path(settings['local_search']['prompt'])
        system=(graph/prompt_path).resolve()
        if not system.is_relative_to(graph.resolve()):raise ValueError('Victim prompt must be inside cloned graph root')
        system.write_text(system.read_text()+'\n'+(ROOT/'prompts/paper_safe.txt').read_text())
    settings['local_search'].update(top_k_entities=c['victim']['top_k_entities'],top_k_relationships=c['victim']['top_k_relationships'],max_context_tokens=c['victim']['context_tokens'])
    settings['models']['default_chat_model'].update(model='${GRAPHRAG_CHAT_MODEL}',max_tokens=c['victim']['output_tokens'],temperature=c['victim']['temperature'],top_p=c['victim']['top_p'])
    for m in settings['models'].values():m.update(max_retries=c['victim']['sdk_max_retries'],request_timeout=c['victim']['request_timeout_seconds'])
    settings['vector_store']['default_vector_store']['db_uri']=str(graph/'output/lancedb')
    for k in ['output','cache','reporting']:settings[k]['base_dir']=str(graph/('logs' if k=='reporting' else k))
    settings.pop('workflows',None)
    (graph/'settings.yaml').write_text(yaml.safe_dump(settings,sort_keys=False))
    write(out/'config.json',c)
    inputs=tree_hash(graph)
    external=Path(c['source_code'])
    from importlib.metadata import version
    versions={name:version(name) for name in ['graphrag','openai','httpx','lancedb','networkx','pyarrow','pandas','sentence-transformers','transformers','torch']}
    write(out/'manifest.json',{'config_sha256':sha(out/'config.json'),'implementation_hashes':implementation_hashes(),'package_versions':versions,
        'external_source':str(external/'src'),'external_hashes':tree_hash(external/'src'),
        'frame_encoder_path':str(encoder_path(c)), 'frame_encoder_hashes':tree_hash(encoder_path(c)),
        'input_hashes':inputs,'source_index_hashes':tree_hash(index),
        'inspection':info,'scope':'Prepared offline; not approved or launched'})
    validate(out)
    write(out/'STATUS.json',{'status':'prepared_awaiting_user_audit','model_calls':0})
    return info


def validate(out):
    out=inside(out);m=read(out/'manifest.json');c=check_config(read(out/'config.json'))
    if sha(out/'config.json')!=m['config_sha256']:raise ValueError('Frozen configuration changed')
    if implementation_hashes()!=m['implementation_hashes']:raise ValueError('Implementation changed after preparation')
    from importlib.metadata import version
    if any(version(name)!=expected for name,expected in m['package_versions'].items()):raise ValueError('Dependency version changed')
    if tree_hash(m['external_source'])!=m['external_hashes']:raise ValueError('Read-only external source changed')
    if tree_hash(m['frame_encoder_path'])!=m['frame_encoder_hashes']:raise ValueError('Sentence encoder weights/configuration changed')
    for name,expected in m['input_hashes'].items():
        if sha(out/'graph_root'/name)!=expected:raise ValueError('Runtime input changed: '+name)
    if tree_hash(out/'graph_root/output')!=m['source_index_hashes']:raise ValueError('Index changed')
    return c,m


def require_approval(out, approval):
    """No file is generated by this code that can authorize its own launch."""
    out=inside(out)
    if approval is None:raise PermissionError('Awaiting user audit; no approval supplied')
    a=read(inside(approval))
    if a.get('approved') is not True or a.get('run')!=str(out) or a.get('manifest_sha256')!=sha(out/'manifest.json'):
        raise PermissionError('Approval must bind this exact prepared run and manifest')
    if not a.get('user_instruction'):raise PermissionError('Missing recorded user authorization')
    return a
