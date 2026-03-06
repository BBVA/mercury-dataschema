import numpy as np
import pandas as pd
import pytest

from mercury.dataschema.calculator import PandasStatCalculator
from mercury.dataschema.feature import (
    BinaryFeature,
    CategoricalFeature,
    ContinuousFeature,
    DataType,
    DiscreteFeature,
    FeatType,
    Feature,
    FeatureFactory,
)


def test_feature_repr_enum_and_json():
    feature = Feature(name='base', dtype=DataType.STRING)
    feature.stats = {'an_int': 1, 'a_bool': True, 'a_float': 1.5}

    assert str(feature) == 'Feature (NAME=base, dtype=DataType.STRING)'
    assert repr(feature) == 'Feature (NAME=base, dtype=DataType.STRING)'
    assert feature.as_enum == FeatType.UNKNOWN

    payload = feature.to_json()
    assert payload['feat_type'] == FeatType.UNKNOWN.value
    assert payload['stats']['a_bool'] is True


def test_specific_feature_classes_repr_enum_and_stats():
    calculator = PandasStatCalculator()

    binary = BinaryFeature(name='b', dtype=DataType.BOOL)
    categorical = CategoricalFeature(name='c', dtype=DataType.STRING)
    discrete = DiscreteFeature(name='d', dtype=DataType.INTEGER)
    continuous = ContinuousFeature(name='x', dtype=DataType.FLOAT)

    assert str(binary) == 'Binary Feature (NAME=b, dtype=DataType.BOOL)'
    assert repr(binary) == 'Binary Feature (NAME=b, dtype=DataType.BOOL)'
    assert binary.as_enum == FeatType.BINARY

    assert str(categorical) == 'Categorical Feature (NAME=c, dtype=DataType.STRING)'
    assert repr(categorical) == 'Categorical Feature (NAME=c, dtype=DataType.STRING)'
    assert categorical.as_enum == FeatType.CATEGORICAL

    built_discrete = discrete.build_stats(pd.Series([1, 2, 2, np.nan]), calculator)
    assert built_discrete.stats['min'] == 1.0
    assert built_discrete.stats['max'] == 2.0
    assert str(discrete) == 'Discrete Feature (NAME=d, dtype=DataType.INTEGER)'
    assert repr(discrete) == 'Discrete Feature (NAME=d, dtype=DataType.INTEGER)'
    assert discrete.as_enum == FeatType.DISCRETE

    assert str(continuous) == 'Continuous Feature (NAME=x, dtype=DataType.FLOAT)'
    assert repr(continuous) == 'Continuous Feature (NAME=x, dtype=DataType.FLOAT)'
    assert continuous.as_enum == FeatType.CONTINUOUS


def test_factory_warning_paths_for_float_and_int_to_categorical():
    factory = FeatureFactory()

    float_column = pd.Series([0.0, 1.0, 2.0, 0.0, 1.0, 2.0], name='float_as_category')
    with pytest.warns(RuntimeWarning, match='FLOAT feature float_as_category converted to Categorical'):
        float_feature = factory.build_feature(
            float_column,
            colname='float_as_category',
            threshold_categorical=0.9,
            verbose=True,
        )
    assert isinstance(float_feature, CategoricalFeature)

    int_column = pd.Series([0, 1, 2, 0, 1, 2], name='int_as_category')
    with pytest.warns(RuntimeWarning, match='INTEGER feature int_as_category converted to Categorical'):
        int_feature = factory.build_feature(
            int_column,
            colname='int_as_category',
            threshold_categorical=0.9,
            verbose=True,
        )
    assert isinstance(int_feature, CategoricalFeature)
