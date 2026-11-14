"""Export only public prompt vocabulary and actually delivered transitions."""
import ast
import hashlib
import json
import re

from learner import canonical, state, validate_public


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def public_prompt(system, user):
    if 'Common valid postcondition structural patterns:' in system:
        raise ValueError('P+ prompt is out of scope')
    config_line = next(line for line in system.splitlines() if line.startswith('Configuration sets: '))
    sets = ast.literal_eval('[' + config_line.split(': ', 1)[1] + ']')
    if set(sets) != {'region_options', 'resource_types', 'tier_options'}:
        raise ValueError('Private set access in prompt')
    if any(x in user for x in ('Valid regions:', 'Rejected regions:', 'Valid cluster tiers:', 'Rejected cluster tiers:')):
        raise ValueError('V+ prompt is out of scope')
    types = dict(re.findall(r'^- `([^`]+)` \((\w+)\)$', system, re.M))
    enum_line = next(line for line in system.splitlines() if line.startswith('Deployment status values: '))
    public = {'variables': types, 'statuses': ast.literal_eval('[' + enum_line.split(': ', 1)[1] + ']')}
    for label, key in [('Resource types', 'resource_types'), ('Region options', 'region_options'),
                       ('Cluster tier options', 'tier_options'), ('Max quota', 'max_quota'),
                       ('Max allocation limit', 'max_allocation_limit')]:
        matches = re.findall(r'^- ' + re.escape(label) + r': (.+)$', user, re.M)
        if len(matches) != 1:
            raise ValueError('Public prompt field missing or duplicated: ' + label)
        public[key] = ast.literal_eval(matches[0])
    validate_public(public)
    return public


def transitions(text):
    marker = 'Concrete counterexamples:\n'
    if marker not in text:
        return []
    body = text.split(marker, 1)[1]
    blocks = re.split(r'^Counterexample \d+ \([^)]+\):\n', body, flags=re.M)
    if len(blocks) == 1:
        raise ValueError('Unrecognized counterexample format')
    observations = []
    for block in blocks[1:]:
        before = re.findall(r'^  Before: (.+)$', block, re.M)
        after = re.findall(r'^  After: (.+)$', block, re.M)
        if len(before) != 1 or len(after) != 1:
            raise ValueError('Counterexample lacks unambiguous Before/After')
        b = json.loads(before[0])
        a = None if after[0] == 'null (sandbox rejected the action)' else json.loads(after[0])
        if a is None and after[0] != 'null (sandbox rejected the action)':
            raise ValueError('Unexplained null')
        state(b)
        if a is not None:
            state(a)
        observations.append({'before': b, 'after': a})
    return observations


def export(record):
    interactions = record['interactions']
    first = interactions[0]['messages']
    if [x['role'] for x in first] != ['system', 'user']:
        raise ValueError('Unexpected initial conversation')
    system = first[0]['content']
    public = public_prompt(system, first[1]['content'])
    accumulated = []
    boundaries = []
    for index, interaction in enumerate(interactions):
        messages = interaction['messages']
        if messages[0] != first[0] or messages[-1]['role'] != 'user':
            raise ValueError('Information policy changed')
        new = transitions(messages[-1]['content'])
        accumulated.extend(new)
        # Only actual requests are walked; rounds/evaluation/future feedback are ignored.
        unique = {canonical(o): o for o in accumulated}
        packet = {'public': public, 'observations': [unique[k] for k in sorted(unique)]}
        boundaries.append({'interaction': index, 'role': interaction['role'],
            'request_messages_sha256': digest(messages), 'new_observations': len(new),
            'delivered_total': len(accumulated), 'unique_observations': len(unique),
            'packet_sha256': digest(packet), 'packet': packet})
    return boundaries
