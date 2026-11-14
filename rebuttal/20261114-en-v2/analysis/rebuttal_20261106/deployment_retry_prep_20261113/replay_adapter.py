"""Explicit budget-follow-up adapter. No CLI launch or network calls on import."""
import copy
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from experiment.deployment.checkpoint import CallJournal, read_sealed, usage_from_dict, ResumeError
from experiment.deployment.initial_candidate import InitialCandidate
from experiment.deployment.runner import DeploymentExperimentRunner
from experiment.deployment.config import DeploymentConfig
from experiment.deployment.information_policy import InformationPolicy
from experiment.modeling import Budgets,Decoding,RunSpec
from experiment.modeling.artifacts import Interaction,sha256_json
import hashlib

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())

class SavedPrefix:
    def __init__(self, source, prefix, delegate):
        self.source=source;self.prefix=prefix;self.delegate=delegate
        self.restored=False;self.checked_boundary=False
        self.prefix_calls=0;self.prefix_tokens=0;self.prefix_latency=0.

    @property
    def replayed_calls(self):return self.prefix_calls+self.delegate.replayed_calls
    @property
    def replayed_tokens(self):return self.prefix_tokens+self.delegate.replayed_tokens
    @property
    def replayed_latency_s(self):return self.prefix_latency+self.delegate.replayed_latency_s

    def invoke(self,session,role,user_text,execute):
        sent=[*session.messages,{'role':'user','content':user_text}]
        if not self.restored:
            old=self.source['interactions'][1];after=self.prefix['after']
            if role!=old['role'] or sent!=old['messages']:
                raise ResumeError('saved prefix prompt mismatch')
            if [x.to_dict() for x in session.interactions]!=after['interactions'][:-1]:
                raise ResumeError('saved prefix initial mismatch')
            expected={k:after['usage'][k]-old['usage'][k] for k in ('prompt_tokens','completion_tokens','total_tokens')}
            if session.ledger.calls!=1 or any(session.ledger.usage.to_dict()[k]!=v for k,v in expected.items()):
                raise ResumeError('saved prefix accounting mismatch')
            assert after['stopped_because'] is None and self.prefix['returned_index']==1
            self.prefix_calls=1;self.prefix_tokens=old['usage']['total_tokens'];self.prefix_latency=old['latency_s'] or 0.
            session.messages=copy.deepcopy(after['messages'])
            session.interactions=[Interaction.from_dict(x) for x in after['interactions']]
            session.ledger.calls=after['calls'];session.ledger.usage=usage_from_dict(after['usage'])
            session.stopped_because=None;self.restored=True
            return session.interactions[1]
        if not self.checked_boundary:
            old=self.source['interactions'][2]
            if role!=old['role'] or sent!=old['messages']:
                raise ResumeError('first expanded request differs beyond budgets')
            self.checked_boundary=True
        return self.delegate.invoke(session,role,user_text,execute)


def prepare_one(job,client,serving,call_folder,stop_requested=None):
    for key in ('source','initial_source','prefix_journal','config'):
        if sha(ROOT/job[key])!=job[key+'_sha256']:raise ResumeError('source/configuration changed')
    r=read(ROOT/job['source']);cfg=read(ROOT/job['config'])
    old=InitialCandidate.load(ROOT/job['initial_source'])
    prefix=read_sealed(ROOT/job['prefix_journal'])
    assert prefix['status']=='completed' and prefix['after']['interactions']==r['interactions'][:2]
    assert serving['max_model_len']==16384 and serving['vllm_version']=='0.19.1'
    assert client.model==r['model'] and client.endpoint==r['endpoint']
    runner=DeploymentExperimentRunner(DeploymentConfig.from_json_file(ROOT/cfg['sandbox_config']),client,
        max_depth=cfg['max_depth'],max_states=cfg['max_states'],config_path=str(ROOT/cfg['sandbox_config']),
        serving_metadata=serving,compact_context=False,context_token_limit=16384,context_margin=256,
        counterexample_limit=5,information_policy=InformationPolicy(**cfg['information_policy']))
    spec=RunSpec(method='active_cegis',seed=r['seed'],decoding=Decoding(temperature=.2,top_p=.95,max_tokens=2048),
                 budgets=Budgets(48,4,65536),use_guided_json=True)
    context=runner.initial_context(spec);previous=old.to_dict()['context']
    changed={k for k in context.keys()|previous.keys() if context.get(k)!=previous.get(k)}
    if changed-{'budgets','context_token_limit','serving'}:raise ResumeError('non-budget initial context changed: '+str(changed))
    # A new wrapper explicitly links the original response; never relabel the old file in place.
    initial=InitialCandidate.create(context,old.interaction,calls=old.calls,usage=old.usage,stopped_because=old.stopped_because)
    provenance={'source':job['source'],'source_sha256':job['source_sha256'],
        'original_initial_sha256':old.sha256,'followup_initial_sha256':initial.sha256,
        'prefix_interactions':2,'first_new_interaction_index':2,
        'original_prefix_journal_sha256':job['prefix_journal_sha256'],
        'logical_prefix_calls':2,'logical_prefix_tokens':prefix['after']['usage']['total_tokens'],
        'reuse_policy':'saved initial plus successful first revision; no prior truncated response replay',
        'retry_adapter_sha256':sha(Path(__file__))}
    delegate=CallJournal(call_folder,sha256_json({'provenance':provenance,'context':context}),stop_requested)
    return runner,spec,initial,SavedPrefix(r,prefix,delegate),provenance
