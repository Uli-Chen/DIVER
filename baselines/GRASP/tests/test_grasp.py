import copy,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from grasp.core import Relation,Parsed,Scheduler,Controller as RealController,parse_reply,policy,regime
from grasp.frames import FrameSelector
from functools import partial

def setUpModule():
    (ROOT / 'runtime').mkdir(parents=True, exist_ok=True)

class FixtureEncoder:
    def encode(self,texts):
        return [[1.0, float(len(x)%7+1)] for x in texts]

def Controller(config):
    return RealController(config,frame_selector=FrameSelector(FixtureEncoder()))
from grasp.prompts import render
from grasp.config import check_config,require_approval
from grasp.io import read,write,sha
from grasp.runner import run


def config():return copy.deepcopy(read(ROOT/'configs/benchmark/novel.json'))


def response(*triples):
    return '[RELATIONS]\n'+'\n'.join(f'- (ID{i}) <{a}> --[<{b}>]--> <{c}>' for i,(a,b,c) in enumerate(triples))+'\n[END RELATIONS]'


class PaperPolicyTests(unittest.TestCase):
    def state(self,ema=0,types=0,zero=0,none=0,last_types=0,last_edges=0):
        s=Scheduler('A',config()['scheduler']);s.ema=ema;s.types={str(i) for i in range(types)}
        s.zero_streak=zero;s.none_streak=none;s.last_types=last_types;s.last_edges=last_edges
        return s

    def test_all_thirteen_table_rows(self):
        cases=[({'none':1},'none1',(.7,0,0,.3)),({'none':2},'none2',(.3,0,0,.7)),
            ({'zero':3},'stall3_scar',(.5,0,.2,.3)),({'zero':3,'types':3},'stall3_rich',(.5,.2,0,.3)),
            ({},'stall_scar',(.3,0,.5,.2)),({'types':3},'stall_rich',(.3,.3,0,.3)),
            ({'ema':3},'surge_scar',(0,0,.5,.5)),({'ema':3,'types':3},'surge_exploit',(0,1,0,0)),
            ({'ema':3,'types':3,'last_types':1},'surge',(0,.5,0,.5)),
            ({'ema':1},'steady_scar',(0,0,1,0)),({'ema':1,'types':3,'last_edges':1},'steady_exploit',(0,.7,0,.3)),
            ({'ema':1,'types':6},'steady_sat',(0,1,0,0)),({'ema':1,'types':3},'steady',(.05,.35,.35,.25))]
        for kw,name,weights in cases:
            with self.subTest(name=name):self.assertEqual(policy(self.state(**kw)),(name,weights))

    def test_priority_over_momentum(self):
        self.assertEqual(policy(self.state(ema=5,types=10,none=2))[0],'none2')

    def test_regime_boundaries(self):
        self.assertEqual([regime(x) for x in [.499,.5,2,2.001]],['stall','steady','steady','surge'])

    def test_good_turing_counts_raw_duplicates(self):
        s=self.state();a=Relation('A','r','B');b=Relation('A','r','C')
        s.window=[[a,a,b],[]]
        self.assertEqual(s.novelty(),.5)

    def test_five_completed_turn_warmup(self):
        s=self.state()
        for _ in range(4):s.observe(Parsed(explicit_none=True),'baseline')
        self.assertIsNone(s.stop_reason())
        s.observe(Parsed(explicit_none=True),'A')
        self.assertEqual(s.stop_reason(),'good_turing')

    def test_diversity_before_stop(self):
        s=self.state();self.assertEqual(s.choose(42,1)['template'],'baseline')
        s.observe(Parsed(explicit_none=True),'baseline')
        d=s.choose(42,2);self.assertIn(d['template'],'AD');self.assertEqual(d['policy'],'none1')

    def test_zero_sum_no_invention_and_weights_normalize(self):
        s=self.state(types=3);s.diversity=True
        d=s.choose(42,2);self.assertAlmostEqual(sum(d['weights'].values()),1)
        self.assertEqual(d['weights']['C'],0)
        self.assertAlmostEqual(d['weights']['A'],1/3)

    def test_ema_soft_reset(self):
        s=self.state();s.observe(Parsed(relations=[Relation('A','r','B')]),'B')
        self.assertEqual(s.ema,.6);self.assertAlmostEqual(s.template_ema['B'],.65)
        self.assertEqual(s.last_types,1)

    def test_per_target_budget(self):
        s=self.state();s.count=10
        self.assertEqual(s.stop_reason(),'target_budget')
        with self.assertRaises(RuntimeError):s.choose(42,11)


class ParserTests(unittest.TestCase):
    def test_direction_target_and_type(self):
        p=parse_reply(response(('B','Treats','A'),('C','visits','D')),'a')
        self.assertEqual(p.relations,[Relation('B','Treats','A')]);self.assertEqual(p.rejected,1)

    def test_repeated_tuple_with_different_ids_retained(self):
        p=parse_reply(response(('A','r','B'),('A','r','B')),'A')
        self.assertEqual(len(p.relations),2)

    def test_conflicting_ids_keep_both(self):
        t=response(('A','r','B'),('A','s','C')).replace('(ID1)','(ID0)')
        p=parse_reply(t,'A');self.assertEqual(p.relations,[Relation('A','r','B'),Relation('A','s','C')]);self.assertEqual(p.rejected,0)

    def test_same_ids_across_turns_not_identity(self):
        a=parse_reply(response(('A','r','B')),'A');b=parse_reply(response(('A','r','C')),'A')
        self.assertNotEqual(a.relations,b.relations)

    def test_none_different_from_failure(self):
        self.assertTrue(parse_reply('[NONE]','A').explicit_none)
        self.assertFalse(parse_reply('I cannot comply','A').explicit_none)

    def test_truncated_closed_block_keeps_complete_records(self):
        self.assertEqual(parse_reply(response(('A','r','B')),'A',truncated=True).relations,[Relation('A','r','B')])

    def test_truncated_open_block_keeps_complete_lines_only(self):
        text='[RELATIONS]\n- (1) A --[r]--> B\n- (2) A --[r]--> INCOMPL'
        p=parse_reply(text,'A',truncated=True)
        self.assertEqual(p.relations,[Relation('A','r','B')]);self.assertEqual(p.status,'partial')

    def test_multiple_blocks_keep_records_and_raw_duplicates(self):
        text=response(('A','r','B'))+'\nCorrection:\n'+response(('A','s','C'),('A','r','B'))
        self.assertEqual(parse_reply(text,'A').relations,[Relation('A','r','B'),Relation('A','s','C'),Relation('A','r','B')])

    def test_multiple_blocks_do_not_parse_prose_between_them(self):
        text=response(('A','r','B'))+'\n- (9) A --[r]--> PROSE\n'+response(('A','s','C'))
        self.assertEqual(parse_reply(text,'A').relations,[Relation('A','r','B'),Relation('A','s','C')])

    def test_invalid_and_complete_records(self):
        t=response(('A','r','B')).replace('[END RELATIONS]','invalid record\n[END RELATIONS]')
        p=parse_reply(t,'A');self.assertEqual(p.status,'partial');self.assertEqual(len(p.relations),1)

    def test_discovery_normalization_and_null(self):
        p=parse_reply('[ENTITIES]\n- < alice >\n- <ALICE>\n- <null>\n[END ENTITIES]',discovery=True)
        self.assertEqual(p.entities,['ALICE']);self.assertEqual(p.rejected,1)

    def test_blank_fields_are_not_entities_or_types(self):
        self.assertEqual(parse_reply(response(('A',' ','B')),'A').relations,[])
        self.assertEqual(parse_reply('[ENTITIES]\n- < >\n[END ENTITIES]',discovery=True).entities,[])

    def test_discovery_omitted_presentation_brackets(self):
        p=parse_reply('[ENTITIES]\n- MEDICAL HISTORY\n- <ALICE>\n[END ENTITIES]',discovery=True)
        self.assertEqual(p.entities,['MEDICAL HISTORY','ALICE']);self.assertEqual(p.rejected,0)

    def test_relations_omitted_presentation_brackets(self):
        p=parse_reply('[RELATIONS]\n- (ID1) ALICE --[knows]--> BOB\n- (ID2) <ALICE> --[visits]--> ROME\n[END RELATIONS]','ALICE')
        self.assertEqual(p.relations,[Relation('ALICE','knows','BOB'),Relation('ALICE','visits','ROME')])

    def test_malformed_brackets_not_repaired(self):
        p=parse_reply('[RELATIONS]\n- (ID1) <ALICE --[knows]--> BOB\n[END RELATIONS]','ALICE')
        self.assertEqual(p.relations,[]);self.assertEqual(p.rejected,1)


class ControlTests(unittest.TestCase):
    def test_discovery_in_budget(self):
        c=config();c['global_budget']=1;x=Controller(c)
        q=x.next_request(render);self.assertEqual(q['kind'],'discovery')
        x.observe({'response':'[ENTITIES]\n- <A>\n[END ENTITIES]'})
        self.assertIsNone(x.next_request(render));self.assertEqual(x.discovery_count,1)

    def test_no_truth_or_unobserved_anchor(self):
        c=config();c['source_graph']='/does/not/exist';c['source_index']='/does/not/exist'
        x=Controller(c);x.next_request(render);x.observe({'response':'[ENTITIES]\n- <A>\n[END ENTITIES]'})
        q=x.next_request(render);self.assertEqual(q['target'],'A')
        self.assertEqual(q['query'],q['retrieval_query']);self.assertEqual(q['query'],q['generation_query'])

    def test_frontier_moves_without_hidden_queries(self):
        c=config();c['scheduler']['per_target_budget']=1;x=Controller(c)
        x.next_request(render);x.observe({'response':'[ENTITIES]\n- <A>\n[END ENTITIES]'})
        x.next_request(render);x.observe({'response':response(('A','r','B'))})
        q=x.next_request(render);self.assertEqual(q['turn'],3);self.assertEqual(q['target'],'B')
        self.assertEqual(x.transitions[0]['reason'],'target_budget')

    def test_native_targeted_does_not_expand(self):
        c=config();c.update(mode='targeted',targets=['A']);c['scheduler']['per_target_budget']=1
        x=Controller(c);x.next_request(render);x.observe({'response':response(('A','r','B'))})
        self.assertIsNone(x.next_request(render))

    def test_query_deterministic(self):
        x,y=Controller(config()),Controller(config())
        for _ in range(7):
            a=x.next_request(render);b=y.next_request(render);self.assertEqual(a,b)
            reply={'response':'[ENTITIES]\n- <A>\n[END ENTITIES]'} if a['kind']=='discovery' else {'response':'[NONE]'}
            self.assertEqual(x.observe(reply),y.observe(reply))

    def test_failure_consumes_budget(self):
        c=config();c.update(mode='targeted',targets=['A'],global_budget=1)
        x=Controller(c);x.next_request(render);result=x.observe({'response':'','error':'Timeout'})
        self.assertEqual(result['status'],'request_error');self.assertEqual(x.turn,1);self.assertIsNone(x.next_request(render))

    def test_length_reply_updates_recovery_but_request_error_still_does_not(self):
        for error,expected in [(None,1),('ProviderRequestFailure',0)]:
            c=config();c.update(mode='targeted',targets=['A'],global_budget=1)
            x=Controller(c);x.next_request(render)
            x.observe({'response':response(('A','r','B')),'finish_reasons':['length'],'error':error})
            self.assertEqual(len(x.edges),expected)

    def test_pending_state_guard(self):
        x=Controller(config());x.next_request(render)
        with self.assertRaises(RuntimeError):x.next_request(render)

    def test_configuration_is_explicit(self):
        c=config();c['scheduler']['novelty_window']=3
        with self.assertRaises(ValueError):check_config(c)


class ReplayTests(unittest.TestCase):
    class FakeVictim:
        calls=0
        def query(self,q):
            self.calls+=1
            if q['kind']=='discovery':return {'response':'[ENTITIES]\n- <A>\n[END ENTITIES]'}
            return {'response':response((q['target'],'r',f'N{q["turn"]}'))}

    def test_fresh_replay_and_tamper(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runtime') as td:
            out=Path(td);c=config();c['global_budget']=12
            v=self.FakeVictim();first=run(c,out,v)
            self.assertEqual(v.calls,12);second=run(c,out,None,True);self.assertEqual(first,second)
            self.assertEqual(read(out/'STATUS.json')['status'],'completed')
            p=out/'receipts/turn_0002.json';x=read(p);x['request']['query']='changed';write(p,x)
            with self.assertRaisesRegex(RuntimeError,'mismatch'):run(c,out,None,True)

    def test_unresolved_inflight_refuses(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runtime') as td:
            out=Path(td);write(out/'receipts/inflight_0001.json',{'turn':1})
            v=self.FakeVictim()
            with self.assertRaisesRegex(RuntimeError,'Unresolved'):run(config(),out,v,True)
            self.assertEqual(v.calls,0)

    def test_resume_from_committed_receipt_no_requery(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runtime') as td:
            out=Path(td);c=config();c['global_budget']=8
            class Interrupt(self.FakeVictim):
                def query(self,q):
                    if q['turn']==4:raise KeyboardInterrupt()
                    return super().query(q)
            with self.assertRaises(KeyboardInterrupt):run(c,out,Interrupt())
            # A real unresolved receipt must be audited; this synthetic failure
            # happened before transport and is resolved explicitly in the test.
            (out/'receipts/inflight_0004.json').unlink()
            v=self.FakeVictim();result=run(c,out,v,True)
            self.assertEqual(v.calls,5);self.assertEqual(result['rounds'],8)

    def test_approval_missing_or_wrong_manifest(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runtime') as td:
            out=Path(td);write(out/'manifest.json',{})
            with self.assertRaises(PermissionError):require_approval(out,None)
            write(out/'approval.json',{'approved':True,'run':str(out),'manifest_sha256':'wrong','user_instruction':'test fixture'})
            with self.assertRaises(PermissionError):require_approval(out,out/'approval.json')


if __name__=='__main__':
    from grasp.boundary import install,kernel
    install(False);kernel(False)
    unittest.main(verbosity=2)
