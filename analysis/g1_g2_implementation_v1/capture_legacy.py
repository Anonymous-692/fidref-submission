"""Read-only snapshot of the legacy request and protocol paths, before the patch."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiment.taubench_retail.config import RetailConfig
from experiment.taubench_retail.dsl import contract_json_schema, VOCABULARY_LEGACY, VOCABULARY_ORDER_NEUTRAL
from experiment.taubench_retail.prompts import system_prompt, direct_prompt
from experiment.taubench_retail.runner import RetailExperimentRunner, METHODS
from experiment.taubench_retail.state import RetailState
from experiment.modeling.runner import RunSpec, Budgets, Decoding


def snapshot():
    config = RetailConfig.from_json(ROOT / 'experiment/configs/taubench_retail_default.json')
    result = {'state': RetailState(order_w1_return_payment_method_id='credit_card_0').to_dict(),
              'schema': contract_json_schema(), 'protocols': {}}
    for protocol in (VOCABULARY_LEGACY, VOCABULARY_ORDER_NEUTRAL):
        runner = RetailExperimentRunner(None, config=config, max_depth=0, max_states=1,
                                        vocabulary_protocol=protocol)
        result['protocols'][protocol] = {
            'system': system_prompt(protocol), 'direct': direct_prompt(config, protocol),
            'hashes': {method: runner.compute_protocol_hash(RunSpec(method=method, seed=0,
                decoding=Decoding(), budgets=Budgets())) for method in METHODS}}
    return result


if __name__ == '__main__':
    print(json.dumps(snapshot(), indent=2))
