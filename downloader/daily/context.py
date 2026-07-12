from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class UpdateContext:
    listed: pd.DataFrame
    query_date: date

    @property
    def listed_codes(self) -> set[str]:
        return set(self.listed["Code"].astype(str))
