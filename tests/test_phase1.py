"""Tests for synthetic data and leakage-aware feature engineering."""

import pandas as pd

from src.data.generate import generate_dataset
from src.preprocessing.prepare import engineer_features


def test_generator_is_reproducible_and_builds_requested_sizes():
    customers_a, transactions_a = generate_dataset(100, 600, 7)
    customers_b, transactions_b = generate_dataset(100, 600, 7)
    assert len(customers_a) == 100
    assert len(transactions_a) == 600
    pd.testing.assert_frame_equal(customers_a, customers_b)
    pd.testing.assert_frame_equal(transactions_a, transactions_b)
    assert transactions_a.fraud.nunique() == 2


def test_feature_engineering_adds_safe_behavior_features():
    row = pd.DataFrame({"timestamp": ["2025-01-04T02:00:00Z"], "amount": [1000],
                        "average_transaction_amount": [500], "distance_from_home": [5]})
    result = engineer_features(row)
    assert result.amount_to_average_ratio.iloc[0] == 2
    assert result.night_transaction.iloc[0] == 1
    assert result.is_weekend.iloc[0] == 1
