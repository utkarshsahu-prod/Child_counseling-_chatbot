from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ConcernClass(str, Enum):
    PRIMARY = "primary_concern"
    DSM_RED_FLAG_CLINICAL = "dsm_red_flag_clinical"
    NON_CLINICAL = "non_clinical"


@dataclass(frozen=True, slots=True)
class ConcernPriorityPolicy:
    class_rank: dict[ConcernClass, int]

    @classmethod
    def default(cls) -> "ConcernPriorityPolicy":
        return cls(
            class_rank={
                ConcernClass.PRIMARY: 0,
                ConcernClass.DSM_RED_FLAG_CLINICAL: 1,
                ConcernClass.NON_CLINICAL: 2,
            }
        )

    def sort_concerns(self, concerns: list[dict]) -> list[dict]:
        """
        Sorts concerns according to policy and stable input order.

        Expected fields:
        - concern_class: one of ConcernClass values
        - discovered_at_turn: int
        """

        def rank(item: dict) -> tuple[int, int]:
            concern_class = ConcernClass(item["concern_class"])
            return (self.class_rank[concern_class], int(item.get("discovered_at_turn", 0)))

        return sorted(concerns, key=rank)


DSM_RED_FLAG_POLICY_TABLE = {
    "must_escalate_immediately": [
        "self_harm_signals",
        "harm_to_others_signals",
        "severe_regression_critical_skills",
    ],
    "clinical_priority_tags": [
        "autism_red_flag",
        "severe_language_delay",
        "developmental_regression",
    ],
}
