import copy,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from grasp.frames import FrameSelector
from grasp.core import Controller,parse_reply,Relation
from grasp.prompts import render
from grasp.config import check_config
from grasp.evaluation import evaluate_targets,sample_targets,typed_truth

def setUpModule():
    (ROOT / 'runtime').mkdir(parents=True, exist_ok=True)

class Encoder:
    def __init__(self,vectors):self.vectors=vectors;self.calls=[]
    def encode(self,texts):self.calls.append(list(texts));return [self.vectors[t] for t in texts]

class SemanticFrameTests(unittest.TestCase):
    def test_three_farthest_use_observed_types_not_seed_or_frame_order(self):
        e=Encoder({'seen':[1,0],'same':[1,0],'opposite':[-1,0],'orthogonal':[0,1],'near':[1,1]})
        s=FrameSelector(e)
        self.assertEqual(s.select(['same','near','orthogonal','opposite'],{'seen'})['frames'],['opposite','orthogonal','near'])
        self.assertEqual(len(e.calls),1)
        s.select(['same','near','orthogonal','opposite'],{'seen'})
        self.assertEqual(len(e.calls),1)
        self.assertEqual(s.select(['same','near','orthogonal','opposite'],{'opposite'})['frames'][0],'same')

    def test_mean_distance_multiple_types_and_deterministic_ties(self):
        e=Encoder({'u':[1,0],'v':[0,1],'f1':[-1,0],'f2':[0,-1],'f3':[1,0],'f4':[1,1]})
        r=FrameSelector(e).select(['f2','f1','f4','f3'],{'v','u'})
        self.assertEqual(r['frames'],['f2','f1','f3']);self.assertAlmostEqual(r['distances'][0],1.5)

    def test_empty_types_and_invalid_vectors(self):
        e=Encoder({});s=FrameSelector(e)
        self.assertEqual(s.select(['one','two','three','four'],[])['frames'],['one','two','three']);self.assertFalse(e.calls)
        with self.assertRaises(ValueError):FrameSelector(Encoder({'x':[0,0],'a':[1,0],'b':[1,0],'c':[1,0]})).select(['a','b','c'],['x'])

    def test_controller_A_prompt_uses_all_three_whole_phrases(self):
        cfg=json.loads((ROOT/'configs/benchmark/novel.json').read_text());cfg.update(mode='targeted',targets=['A'])
        cfg['frames']=['Same frame','Opposite frame','Orthogonal frame','Nearby frame']
        e=Encoder({'seen':[1,0],'Same frame':[1,0],'Opposite frame':[-1,0],'Orthogonal frame':[0,1],'Nearby frame':[1,1]})
        c=Controller(cfg,FrameSelector(e));c.next_request(render)
        c.observe({'response':'[RELATIONS]\n- (1) A --[seen]--> B\n[END RELATIONS]'})
        with patch.object(c.active,'choose',return_value={'template':'A'}):q=c.next_request(render)
        self.assertEqual(q['frame_hints'],['Opposite frame','Orthogonal frame','Nearby frame'])
        self.assertIn('{Opposite frame, Orthogonal frame, Nearby frame}',q['query'])
        self.assertNotIn('{FRAME_HINTS}',q['query'])

class ParserFeedbackTests(unittest.TestCase):
    def test_none_in_closed_block_or_terminal_prose(self):
        for text in ['[NONE]','No matching rows.\n[NONE]','Explanation.\n[RELATIONS]\n[NONE]\n[END RELATIONS]']:
            self.assertTrue(parse_reply(text,'A').explicit_none)
        self.assertFalse(parse_reply('The token [NONE] is a format example.','A').explicit_none)

    def test_none_does_not_erase_relations_or_truncation(self):
        text='[RELATIONS]\n- (1) A --[r]--> B\n[END RELATIONS]\n[RELATIONS]\n[NONE]\n[END RELATIONS]'
        p=parse_reply(text,'A');self.assertFalse(p.explicit_none);self.assertEqual(p.relations,[Relation('A','r','B')])
        self.assertFalse(parse_reply('[RELATIONS]\n[NONE]\n[END RELATIONS]','A',truncated=True).explicit_none)

    def test_wrapped_none_updates_scheduler_priority(self):
        cfg=json.loads((ROOT/'configs/benchmark/novel.json').read_text());cfg.update(mode='targeted',targets=['A'])
        c=Controller(cfg,FrameSelector(Encoder({})));c.next_request(render)
        c.observe({'response':'[RELATIONS]\n[NONE]\n[END RELATIONS]'})
        self.assertEqual(c.active.none_streak,1);self.assertEqual(c.active.choose(42,2)['policy'],'none1')

    def test_verbatim_preserves_case_and_internal_spacing(self):
        p=parse_reply('[RELATIONS]\n- (1) Alice --[is  Friend]--> Bob\n[END RELATIONS]','Alice',verbatim=True)
        self.assertEqual(p.relations,[Relation('Alice','is  Friend','Bob')])
        self.assertFalse(parse_reply('[RELATIONS]\n- (1) ALICE --[r]--> Bob\n[END RELATIONS]','Alice',verbatim=True).relations)

class TargetEvaluationTests(unittest.TestCase):
    def test_types_direction_and_false_predictions_are_counted(self):
        truth={('A','r','B'),('A','s','C'),('X','q','Y')}
        preds={'A':{('A','wrong','B'),('C','s','A'),('A','r','Z')},'X':{('X','q','Y')}}
        result=evaluate_targets(truth,['A','X'],preds)
        self.assertEqual(result['targets']['A']['RType']['matched'],0)
        self.assertEqual(result['targets']['A']['Naive']['matched'],1)
        self.assertEqual(result['targets']['A']['Naive']['predicted'],3)
        self.assertEqual(result['macro']['RType']['precision'],.5)
        self.assertAlmostEqual(result['macro']['Naive']['precision'],2/3)

    def test_degree_sampling_is_offline_and_excludes_small_targets(self):
        t={('A','r',str(i)) for i in range(5)}|{('X','s','Y')}
        self.assertEqual(sample_targets(t,42,count=1),['A'])
        with self.assertRaises(ValueError):sample_targets(t,42,count=2)

    def test_missing_type_column_rejected(self):
        import pandas as pd
        with tempfile.TemporaryDirectory(dir=ROOT/'runtime') as td:
            pd.DataFrame([{'source':'A','target':'B','description':'r'}]).to_parquet(Path(td)/'relationships.parquet')
            with self.assertRaisesRegex(ValueError,'explicit'):typed_truth(td,'relation_type')

    def test_paper_profile_fixed_targets_no_expansion(self):
        cfg=json.loads((ROOT/'configs/paper_targeted.example.json').read_text());cfg['targets']=['Alice','Zed'];check_config(cfg)
        c=Controller(cfg,FrameSelector(Encoder({})));q=c.next_request(render);self.assertEqual(q['target'],'Alice')
        c.observe({'response':'[RELATIONS]\n- (1) Alice --[knows]--> New Person\n[END RELATIONS]'})
        self.assertEqual(list(c.queue),['Zed']);self.assertIn('New Person',c.nodes)
        cfg['victim']['output_tokens']=16384
        with self.assertRaises(ValueError):check_config(cfg)

    def test_targeted_run_replay_and_macro_evaluation(self):
        import pandas as pd
        from grasp.runner import run,offline_evaluate
        cfg=json.loads((ROOT/'configs/paper_targeted.example.json').read_text());cfg['targets']=['Alice','Zed']
        frames=cfg['frames'];encoder=Encoder({**{f:[1,1] for f in frames},'knows':[1,0]})
        class Victim:
            def query(self,q):
                if q['episode_round']==1:
                    return {'response':f"[RELATIONS]\n- (1) {q['target']} --[knows]--> Bob\n[END RELATIONS]"}
                return {'response':'[RELATIONS]\n[NONE]\n[END RELATIONS]'}
        with tempfile.TemporaryDirectory(dir=ROOT/'runtime') as td:
            out=Path(td);truth=out/'graph_root/output';truth.mkdir(parents=True)
            pd.DataFrame([{'source':t,'relation_type':'knows','target':'Bob'} for t in cfg['targets']]).to_parquet(truth/'relationships.parquet')
            cfg['source_graph']=str(truth.parent)
            with patch('grasp.frames.selector_from_config',side_effect=lambda c:FrameSelector(encoder)):
                first=run(cfg,out,Victim());second=run(cfg,out,None,True)
            self.assertEqual(first,second);self.assertEqual(first['discovery_rounds'],0)
            self.assertEqual(offline_evaluate(cfg,out)['macro']['RType']['f1'],1.0)

if __name__=='__main__':unittest.main()
