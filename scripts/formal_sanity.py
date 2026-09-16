#!/usr/bin/env python3
"""Enumerate binary joint laws for the temporal-validity sanity check."""

from __future__ import annotations

import json
from pathlib import Path


DENOMINATOR = 20


def main() -> None:
    violations_protocol_bound = []
    violations_sharp_binary_bound = []
    attainable: dict[tuple[int, int], set[int]] = {}

    # Cells are counts for (D, E) in {(0,0), (0,1), (1,0), (1,1)}, where
    # D = 1[Y_old != Y_cur] and E = 1[prediction != Y_old]. For binary
    # labels, current error is 1[D != E].
    n = DENOMINATOR
    for n00 in range(n + 1):
        for n01 in range(n - n00 + 1):
            for n10 in range(n - n00 - n01 + 1):
                n11 = n - n00 - n01 - n10
                e_old_n = n01 + n11
                delta_n = n10 + n11
                e_cur_n = n01 + n10
                attainable.setdefault((e_old_n, delta_n), set()).add(e_cur_n)

                protocol_lo = abs(e_old_n - delta_n)
                protocol_hi = min(e_old_n + delta_n, n)
                sharp_hi = min(e_old_n + delta_n, 2 * n - e_old_n - delta_n)
                if not protocol_lo <= e_cur_n <= protocol_hi:
                    violations_protocol_bound.append([n00, n01, n10, n11])
                if not protocol_lo <= e_cur_n <= sharp_hi:
                    violations_sharp_binary_bound.append([n00, n01, n10, n11])

    non_identifiable_pairs = []
    for (e_old_n, delta_n), values in sorted(attainable.items()):
        if len(values) > 1:
            non_identifiable_pairs.append(
                {
                    "e_old": e_old_n / n,
                    "delta": delta_n / n,
                    "e_cur_min": min(values) / n,
                    "e_cur_max": max(values) / n,
                    "attainable_count": len(values),
                }
            )

    construction = {
        "shared_marginals": {"e_old": 0.25, "delta": 0.25},
        "law_a": {
            "description": "D와 E가 같은 확률 0.25 사건에서 함께 발생",
            "joint_probabilities_D_E": {
                "00": 0.75,
                "01": 0.0,
                "10": 0.0,
                "11": 0.25,
            },
            "e_cur": 0.0,
        },
        "law_b": {
            "description": "D와 E가 서로 겹치지 않는 확률 0.25 사건에서 발생",
            "joint_probabilities_D_E": {
                "00": 0.5,
                "01": 0.25,
                "10": 0.25,
                "11": 0.0,
            },
            "e_cur": 0.5,
        },
    }

    output = {
        "denominator": n,
        "joint_laws_enumerated": sum(len(v) for v in attainable.values()),
        "distinct_marginal_pairs": len(attainable),
        "non_identifiable_marginal_pairs": len(non_identifiable_pairs),
        "protocol_bound_violations": len(violations_protocol_bound),
        "sharp_binary_bound_violations": len(violations_sharp_binary_bound),
        "protocol_bound": "|e_old-delta| <= e_cur <= min(e_old+delta, 1)",
        "sharp_binary_bound": (
            "|e_old-delta| <= e_cur <= "
            "min(e_old+delta, 2-e_old-delta)"
        ),
        "explicit_non_identifiability_construction": construction,
        "sample_non_identifiable_pairs": non_identifiable_pairs[:10],
    }

    out_path = Path(__file__).resolve().parents[1] / "results" / "formal_sanity.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")

    assert not violations_protocol_bound
    assert not violations_sharp_binary_bound
    assert construction["law_a"]["e_cur"] != construction["law_b"]["e_cur"]
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
