import pytest
import pandas as pd
import numpy as np

from mercury.dataschema.calculator import StatCalculatorFactory, PandasStatCalculator
from mercury.dataschema.feature import Feature


@pytest.fixture(scope='module')
def pandas_df():
    data = [['tom', 10], ['nick', 15], ['juli', 14]]
    return pd.DataFrame(data, columns=['Name', 'Age'])


def test_calculator_factory(pandas_df):
    assert isinstance(StatCalculatorFactory.build_calculator(pandas_df), PandasStatCalculator)


def test_calculator(pandas_df):
    calculator = StatCalculatorFactory.build_calculator(pandas_df)

    feature = Feature()

    calculator.min(pandas_df['Age'], feature)
    calculator.max(pandas_df['Age'], feature)
    calculator.std(pandas_df['Age'], feature)
    calculator.mean(pandas_df['Age'], feature)

    assert feature.stats['min'] == 10
    assert feature.stats['max'] == 15
    assert feature.stats['mean'] == 13


def test_set_config(pandas_df):
    calculator = StatCalculatorFactory.build_calculator(pandas_df)
    with pytest.raises(ValueError):
        calculator.set_config(**{'nonexistingattr': 10})

    # assert it assigns the property well
    calculator.set_config(**{'distribution_bins_method': 10})
    assert calculator.distribution_bins_method == 10

    # Assert does nothing with None
    calculator.set_config()
