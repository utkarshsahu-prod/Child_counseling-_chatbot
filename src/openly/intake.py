from __future__ import annotations

from dataclasses import dataclass, field


REQUIRED_PRIMARY_INTAKE_FIELDS = (
    "frequency",
    "intensity",
    "current_methods",
    "where_happening",
    "life_impact",
)


@dataclass(slots=True)
class PrimaryConcernIntake:
    fields: dict[str, str] = field(default_factory=dict)

    def update(self, payload: dict[str, str]) -> None:
        for key, value in payload.items():
            normalized_key = str(key).strip().lower()
            if normalized_key not in REQUIRED_PRIMARY_INTAKE_FIELDS:
                continue

            normalized_value = str(value).strip()
            if not normalized_value:
                continue

            self.fields[normalized_key] = normalized_value

    def missing_fields(self) -> list[str]:
        return [name for name in REQUIRED_PRIMARY_INTAKE_FIELDS if name not in self.fields]

    @property
    def is_complete(self) -> bool:
        return not self.missing_fields()

    def as_dict(self) -> dict[str, str]:
        return {k: self.fields[k] for k in REQUIRED_PRIMARY_INTAKE_FIELDS if k in self.fields}
