from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .contracts import validate_contract_freeze
from .runtime import build_orchestrator_from_excels


@dataclass(frozen=True, slots=True)
class ReadinessGateResult:
    gate: str
    passed: bool
    details: str


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    passed: bool
    gates: list[ReadinessGateResult]


def run_readiness_review(
    *,
    domain_tree_path: Path | str,
    cross_domain_path: Path | str,
    executed_test_count: int,
    minimum_test_count: int = 25,
) -> ReadinessReport:
    gates: list[ReadinessGateResult] = []

    contract_result = validate_contract_freeze()
    gates.append(
        ReadinessGateResult(
            gate="contracts_frozen",
            passed=contract_result.is_valid,
            details="ok" if contract_result.is_valid else ";".join(contract_result.errors),
        )
    )

    try:
        _orchestrator, assets = build_orchestrator_from_excels(domain_tree_path, cross_domain_path)
        asset_ok = assets.domain_question_count > 0 and assets.severity_rule_count > 0
        gates.append(
            ReadinessGateResult(
                gate="runtime_assets_loaded",
                passed=asset_ok,
                details=f"domain_questions={assets.domain_question_count},severity_rules={assets.severity_rule_count}",
            )
        )
    except Exception as exc:  # noqa: BLE001
        gates.append(
            ReadinessGateResult(
                gate="runtime_assets_loaded",
                passed=False,
                details=f"error:{type(exc).__name__}:{exc}",
            )
        )

    tests_ok = executed_test_count >= minimum_test_count
    gates.append(
        ReadinessGateResult(
            gate="test_coverage_threshold",
            passed=tests_ok,
            details=f"executed={executed_test_count},minimum={minimum_test_count}",
        )
    )

    return ReadinessReport(passed=all(g.passed for g in gates), gates=gates)
