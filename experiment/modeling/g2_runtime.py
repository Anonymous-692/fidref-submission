"""Opt-in Deployment replay/accounting policy; legacy B16 does not use this module."""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .client import Usage
from .st2x2 import (batch_size_for, SELECTION_PARTITIONED, SELECTION_UNIFORM,
                    TERMINATION_EXHAUST, TERMINATION_STOP)

VERSION = "g2_replay_accounting_v2"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def configure(runner, window, margin):
    if window is None or window < 1 or margin < 0 or margin >= window:
        raise ValueError("G2 requires an explicit context window and a smaller nonnegative margin")
    runner.context_token_limit = window
    runner.context_margin = margin
    root = Path(__file__).resolve().parents[2]
    sources = ['scripts/run_st2x2.py', 'experiment/modeling/g2_runtime.py',
               'experiment/modeling/st2x2.py', 'experiment/modeling/st2x2_domains.py',
               'experiment/modeling/client.py', 'experiment/deployment/runner.py',
               'experiment/deployment/prompts.py', 'experiment/deployment/dsl.py',
               'experiment/deployment/config.py', 'experiment/deployment/state.py',
               'experiment/deployment/ground_truth.py', 'experiment/deployment/contracts.py',
               'experiment/deployment/env.py', 'experiment/deployment/enumeration.py']
    runner.g2_runtime = {
        'version': VERSION, 'context_window': window, 'safety_margin': margin,
        'compact_context': runner.compact_context,
        'token_policy': 'initial_and_actual_usage_with_prompt_reservation_v1',
        'missing_initial': 'terminal_no_redraw',
        'unknown_usage': 'stop_and_preserve',
        'sources_sha256': {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sources},
    }


def usage_valid(usage):
    keys = ('prompt_tokens', 'completion_tokens', 'total_tokens')
    return (isinstance(usage, dict)
            and all(type(usage.get(k)) is int and usage[k] >= 0 for k in keys)
            and usage['prompt_tokens'] > 0
            and usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens'])


def record_digest(entry):
    return digest({k: v for k, v in entry.items() if k != 'record_sha256'})


def validate_initial(entry):
    if entry.get('record_sha256') != record_digest(entry):
        raise ValueError('G2 initial record hash mismatch')
    if type(entry.get('usage_known')) is not bool:
        raise ValueError('G2 initial record requires explicit usage status')
    if entry['usage_known'] and not usage_valid(entry.get('usage')):
        raise ValueError('G2 known initial usage is invalid')
    if entry.get('source') is not None:
        if not entry.get('usage_known') or not usage_valid(entry.get('usage')):
            raise ValueError('G2 usable initial source requires complete token usage')


def initial_metadata(session):
    inter = session.interactions[-1] if session.interactions else None
    # The raw block, not Usage.from_payload's zero defaults, certifies known usage.
    raw = getattr(inter, 'raw_response', None) or {}
    known = usage_valid(raw.get('usage'))
    return {'usage_known': known, 'initial_stop': session.stopped_because or None,
            'initial_response_text': getattr(inter, 'response_text', None),
            'initial_error': getattr(inter, 'error', None),
            'initial_call_slots_used': session.ledger.calls,
            'initial_interaction': asdict(inter) if inter is not None else None}


def replay_initial(session, entry):
    validate_initial(entry)
    session.ledger.calls += 1
    if entry.get('usage_known') and usage_valid(entry.get('usage')):
        session.ledger.usage = session.ledger.usage + Usage.from_payload(entry['usage'])


def missing_result(entry, *, state_budget, query_budget, balance, exhaust):
    return {'rounds': [], 'parsed': None, 'source': None,
            'stopped_because': entry.get('initial_stop') or 'initial_response_missing',
            'model_calls': 1, 'states_observed': 0,
            'batch_size': batch_size_for(state_budget, query_budget),
            'selection': SELECTION_PARTITIONED if balance else SELECTION_UNIFORM,
            'termination': TERMINATION_EXHAUST if exhaust else TERMINATION_STOP}


def sync_ledger(session, result):
    audits = [r for r in result['rounds'] if 'report' in r]
    session.ledger.states_observed = result['states_observed']
    session.ledger.sampled_states_checked = sum(r['report']['states_checked'] for r in audits)
    session.ledger.sampled_feedback_queries = len(audits)


def accounting(session, entry, result):
    return {'version': VERSION, 'initial_record_sha256': entry['record_sha256'],
            'initial_usage_known': entry['usage_known'],
            'usage_complete': entry['usage_known'] and all(
                not i.error and usage_valid((i.raw_response or {}).get('usage'))
                for i in session.interactions if i.response_text is not None or i.error),
            'initial_call_charged': 1,
            'initial_tokens_charged': entry['usage']['total_tokens'] if entry['usage_known'] else None,
            'unique_audited_states': result['states_observed'],
            'cumulative_state_rescorings': session.ledger.sampled_states_checked,
            'audit_reports': session.ledger.sampled_feedback_queries,
            'public_revision_requests': sum(i.role.startswith('revision_') for i in session.interactions),
            'loop_call_slots': result['model_calls'],
            'note': 'Candidate enumeration, target executions and physical shared-generation cost are separate; unknown usage is not zero cost.'}
