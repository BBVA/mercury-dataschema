import pytest
import pandas as pd
import numpy as np

from mercury.dataschema.calculator import StatCalculatorFactory, PandasStatCalculator
from mercury.dataschema.calculator import FeatureCalculator, SparkStatCalculator
from mercury.dataschema.feature import ContinuousFeature, DataType, Feature


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


def test_feature_calculator_base_noops():
    calculator = FeatureCalculator()
    feature = Feature()

    calculator._FeatureCalculator__init()
    assert calculator.min([], feature) is None
    assert calculator.max([], feature) is None
    assert calculator.distribution([], feature) is None


def test_distribution_cache_miss_and_spark_paths():
    calculator = PandasStatCalculator()
    feature = ContinuousFeature(name='value', dtype=DataType.FLOAT)
    column = pd.Series([1.0, 2.0, np.nan, 3.0])

    calculator.distribution(column, feature, bins=2)

    assert 'no_nan_filtered' in feature.cache
    assert len(feature.stats['distribution']) == 2

    spark_calc = SparkStatCalculator()
    assert isinstance(spark_calc, SparkStatCalculator)

    FakeSparkDataFrame = type('DataFrame', (), {})
    FakeSparkDataFrame.__module__ = 'pyspark.sql'

    with pytest.raises(RuntimeError, match='Pyspark is not supported yet'):
        StatCalculatorFactory.build_calculator(FakeSparkDataFrame())
