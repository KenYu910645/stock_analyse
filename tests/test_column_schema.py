from __future__ import annotations

from column_schema import (
    CHINESE_TO_CANONICAL,
    COLUMN_TRANSLATIONS,
    canonical_name,
    storage_name,
)


def test_column_translations_have_unique_storage_names() -> None:
    storage_names = list(COLUMN_TRANSLATIONS.values())

    assert len(storage_names) == len(set(storage_names))
    assert len(CHINESE_TO_CANONICAL) == len(COLUMN_TRANSLATIONS)


def test_column_translations_round_trip() -> None:
    for canonical, storage in COLUMN_TRANSLATIONS.items():
        assert storage_name(canonical) == storage
        assert canonical_name(storage) == canonical


def test_taifex_open_interest_columns_remain_distinct() -> None:
    assert storage_name("open_interest") == "期貨未平倉量"
    assert storage_name("OI") == "選擇權未平倉量"
