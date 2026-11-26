from __future__ import annotations

from pathlib import Path

from backend.app.core.parser import extract_clean_merchant, parse_alfa_statement


def test_extract_clean_merchant() -> None:
    line = "548673++++++1234    482910\\RUS\\MOSCOW\\CLOUDPAYMENTS*IP IVANOV     15.10.2026 990.00 RUR MCC8299"
    assert extract_clean_merchant(line, is_sbp=False) == "CLOUDPAYMENTS*IP IVANOV"


def test_parse_sample_statement() -> None:
    sample_path = Path("data/sample_alfa_statement.csv")
    assert sample_path.exists(), "Сначала запустите ml_pipeline/build_dataset.py"

    result = parse_alfa_statement(sample_path)
    assert result.expense_rows > 0
    assert result.masking_stats.cards_masked > 0
    assert result.masking_stats.persons_masked > 0
    assert "merchant_clean" in result.df.columns
    assert "Описание_masked" in result.df.columns
