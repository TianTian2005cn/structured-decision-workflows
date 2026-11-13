"""Validated record schema. Gold labels are used ONLY by evaluation."""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Any
import json


@dataclass
class Sample:
    id: str
    task: str
    text: str
    options: dict[str, str]
    gold: frozenset[str]
    split: str = "test"
    requires: dict[str, list[str]] = field(default_factory=dict)
    excludes: list[list[str]] = field(default_factory=list)
    costs: dict[str, float] = field(default_factory=dict)
    budget: float | None = None
    pair_weights: dict[str, float] = field(default_factory=dict)
    mock_scores: dict[str, float] = field(default_factory=dict)
    mock_cardinality: int | None = None
    selected_so_far: tuple[str, ...] = ()
    requires_any: dict[str, list[str]] = field(default_factory=dict)
    precedes: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "Sample":
        required = {"id", "task", "text", "options", "gold"}
        missing = required - item.keys()
        if missing:
            raise ValueError(f"Missing keys: {sorted(missing)}")
        opts = item["options"]
        if not isinstance(opts, dict) or not opts or len(opts) != len(set(opts)):
            raise ValueError("options must be a nonempty dictionary with unique identifiers")
        labels = set(opts)
        gold = frozenset(item["gold"])
        if not gold <= labels:
            raise ValueError(f"gold not in option identifiers: {gold-labels}")
        requires = item.get("requires", {})
        requires_any = item.get('requires_any', {})
        precedes = item.get('precedes', {})
        excludes = item.get("excludes", [])
        if any(k not in labels or any(x not in labels for x in v) for k, v in requires.items()):
            raise ValueError("Unknown option in requires")
        for name,rules in [('requires_any',requires_any),('precedes',precedes)]:
            if any(k not in labels or any(x not in labels for x in v) for k,v in rules.items()):
                raise ValueError('Unknown option in '+name)
        if any(not values for values in requires_any.values()):
            raise ValueError('requires_any alternatives must be nonempty')
        if any(len(pair) != 2 or any(x not in labels for x in pair) for pair in excludes):
            raise ValueError("Invalid exclusion pair")
        scores = item.get("mock_scores", {})
        if any(k not in labels or not 0 <= v <= 1 for k,v in scores.items()):
            raise ValueError("Invalid mock scores")
        costs = {k: float(v) for k,v in item.get("costs", {}).items()}
        if any(k not in labels or v < 0 for k,v in costs.items()):
            raise ValueError("Invalid costs")
        budget = item.get("budget")
        if budget is not None and float(budget) < 0:
            raise ValueError("budget must be nonnegative")
        return cls(id=str(item['id']), task=str(item['task']), text=str(item['text']),
                   options={str(k): str(v) for k,v in opts.items()}, gold=gold,
                   split=item.get('split','test'),
                   requires={k: list(v) for k,v in requires.items()},
                   excludes=[list(pair) for pair in excludes], costs=costs,
                   budget=None if budget is None else float(budget),
                   pair_weights={k: float(v) for k,v in item.get('pair_weights',{}).items()},
                   mock_scores={k: float(v) for k,v in scores.items()},
                   mock_cardinality=item.get('mock_cardinality'),
                   requires_any={k:list(v) for k,v in requires_any.items()},
                   precedes={k:list(v) for k,v in precedes.items()})

    def public_state(self) -> dict[str, Any]:
        """No gold, no mock scores, no gold-derived metadata transmitted to Jev."""
        result = {"task": self.task, "input": self.text,
                  "options": [{"id":k,"meaning":v} for k,v in self.options.items()]}
        if self.selected_so_far:
            result['already_selected'] = list(self.selected_so_far)
        if self.requires:
            result['requires'] = self.requires
        if self.requires_any:
            result['requires_any'] = self.requires_any
        if self.precedes:
            result['precedes_if_both_selected'] = self.precedes
        if self.excludes:
            result['excludes'] = self.excludes
        if self.budget is not None:
            result['costs'] = self.costs
            result['budget'] = self.budget
        return result

    def with_selected(self, picked: set[str]) -> "Sample":
        return replace(self, selected_so_far=tuple(sorted(picked)))


def read_jsonl(path: str) -> list[Sample]:
    records = []
    with open(path, encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    records.append(Sample.from_dict(json.loads(line)))
                except Exception as exc:
                    raise ValueError(f'{path}:{line_no}: {exc}') from exc
    if not records:
        raise ValueError('Empty dataset')
    if len({s.id for s in records}) != len(records):
        raise ValueError('Duplicate record identifiers')
    return records
