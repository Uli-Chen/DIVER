"""Terminal-response accounting regressions; no HTTP transport is used."""
import json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from grasp.io import read
from grasp.transport import NativeVictim

def setUpModule():
    (ROOT / 'runtime').mkdir(parents=True, exist_ok=True)

class AccountingTests(unittest.TestCase):
    def run_records(self,rows,answer):
        with tempfile.TemporaryDirectory(dir=ROOT/'runtime') as td:
            victim=NativeVictim.__new__(NativeVictim)
            victim.c=read(ROOT/'configs/benchmark/novel.json');victim.out=Path(td);victim.runtime=None
            victim.audit=SimpleNamespace(count=0,path=Path(td)/'requests.jsonl',turn=0)
            def query(**kw):
                victim.audit.path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
                victim.audit.count=len(rows)
                return answer,{'hidden_context':'must not be returned'}
            victim.query_fn=query
            return victim.query({'turn':1,'generation_query':'synthetic test'})

    def test_failed_attempt_then_success_retains_answer_and_all_costs(self):
        rows=[{'kind':'embedding','error':'Timeout','status':None,'usage_complete':False},
              {'kind':'embedding','status':200,'usage_complete':True},
              {'kind':'chat','status':200,'finish_reasons':['stop'],'usage_complete':True}]
        reply=self.run_records(rows,'[NONE]')
        self.assertEqual(reply['response'],'[NONE]');self.assertIsNone(reply['error'])
        self.assertEqual(reply['request_count'],3);self.assertEqual(reply['usage_unknown'],1)
        self.assertNotIn('context',reply)

    def test_last_chat_controls_truncation(self):
        rows=[{'kind':'chat','status':200,'finish_reasons':['length'],'usage_complete':True},
              {'kind':'chat','status':200,'finish_reasons':['stop'],'usage_complete':True}]
        self.assertEqual(self.run_records(rows,'[NONE]')['finish_reasons'],['stop'])

    def test_terminal_failure_is_not_success(self):
        reply=self.run_records([{'kind':'chat','status':500,'usage_complete':False}],'')
        self.assertEqual(reply['error'],'ProviderRequestFailure')
        self.assertEqual(reply['request_count'],1)

if __name__=='__main__':
    from grasp.boundary import install,kernel
    install(False);kernel(False)
    unittest.main(verbosity=2)
