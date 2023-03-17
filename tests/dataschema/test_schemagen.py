import pytest

import seaborn as sns
import pandas as pd
import numpy as np

from mercury.dataschema.feature import (
    CategoricalFeature,
    BinaryFeature,
    ContinuousFeature,
    DiscreteFeature,
    DataType
)
from mercury.dataschema import DataSchema
from mercury.dataschema.feature import DataType, FeatType
from mercury.dataschema.anonymize import Anonymize


@pytest.fixture(scope='module')
def datasets():
    tips = sns.load_dataset('tips')
    tips['sex'] = tips['sex'].astype(str)
    tips['smoker'] = tips['smoker'].astype(str)
    tips['day'] = tips['day'].astype(str)
    tips['time'] = tips['time'].astype(str)

    titanic = sns.load_dataset('titanic')
    isna_deck = titanic.deck.isna()
    titanic['class'] = titanic['class'].astype(str)
    titanic['deck'] = titanic['deck'].astype(str)
    titanic['who'] = titanic['who'].astype(str)
    titanic.loc[isna_deck, 'deck'] = np.nan

    return tips, titanic


def test_dataschema_build(datasets):
    tips, titanic = datasets

    schma = DataSchema().generate(tips)

    assert isinstance(schma.feats['sex'], BinaryFeature)
    assert isinstance(schma.feats['smoker'], BinaryFeature)
    assert isinstance(schma.feats['time'], BinaryFeature)
    assert isinstance(schma.feats['size'], CategoricalFeature)
    assert isinstance(schma.feats['day'], CategoricalFeature)
    assert isinstance(schma.feats['total_bill'], ContinuousFeature)
    assert isinstance(schma.feats['tip'], ContinuousFeature)

    assert schma.feats['sex'].name == 'sex'
    assert schma.feats['size'].name == 'size'
    assert schma.feats['total_bill'].name == 'total_bill'

    schma = DataSchema().generate(titanic)
    assert isinstance(schma.feats['deck'], CategoricalFeature)
    assert(schma.feats['deck'].stats['percent_nan'] > 0)
    assert(schma.feats['adult_male'].dtype == DataType.BOOL)
    assert schma.feats['adult_male'].name == 'adult_male'


def test_dataschema_stats(datasets):
    tips, titanic = datasets

    schma = DataSchema().generate(tips).calculate_statistics()

    assert schma.feats['tip'].stats['min'] == 1.0
    assert schma.feats['tip'].stats['max'] == 10.0
    assert schma.feats['tip'].stats['mean'] == pytest.approx(2.99827868852459)
    assert schma.feats['tip'].stats['percent_unique'] == pytest.approx(0.5040983606557377)

    schma = DataSchema().generate(titanic).calculate_statistics()
    assert schma.feats['sex'].stats['distribution_bins'][0] == 'female'
    assert schma.feats['sex'].stats['distribution_bins'][1] == 'male'
    assert schma.feats['sex'].stats['distribution'][0] == pytest.approx(0.35655738, 0.1)


def test_dataschema_stats_custom_params(datasets):
    _, titanic = datasets
    schma = DataSchema().generate(titanic).calculate_statistics()
    assert len(schma.feats['age'].stats['distribution']) > 10

    schma = DataSchema().generate(titanic).calculate_statistics({'distribution_bins_method': 5})
    assert len(schma.feats['age'].stats['distribution']) == 5

    schma = DataSchema().generate(titanic).calculate_statistics({
        'age': {'distribution_bins_method': 5},
        'fare': {'distribution_bins_method': 3}
    })
    assert len(schma.feats['age'].stats['distribution']) == 5
    assert len(schma.feats['fare'].stats['distribution']) == 3

    titanic = titanic.reset_index().rename(columns = {'index':'ID'})
    titanic["ID"] = titanic["ID"].astype(str)
    schma = (DataSchema().generate(titanic).calculate_statistics({'limit_categorical_perc': 0.05}))
    assert len(schma.feats['ID'].stats['domain']) == 44

    with pytest.raises(ValueError) as e:
        schma = (DataSchema().generate(titanic).calculate_statistics({'limit_categorical_perc': 5}))
        assert "Input Error: 'limit_categorical_perc' must be a float between 0 and 1" in str(e.value)


def test_dataschema_stats_anonymize(datasets, tmpdir):
    
    _, titanic = datasets
    test_feat = "class"

    #Original
    schema_orig = DataSchema().generate(titanic, verbose = False).calculate_statistics()
    assert sorted(schema_orig.feats[test_feat].stats['domain']) == ['First', 'Second', 'Third']

    #Anonymized
    an_encrypt = Anonymize(0)
    an_encrypt.set_key("07jaPY")
    anon_dict = {test_feat : an_encrypt}
    schma_an = schema_orig.anonymize(anonymize_params=anon_dict)
    an_domain = schma_an.feats[test_feat].stats['domain']
    an_dist_bins = schma_an.feats[test_feat].stats['distribution_bins']
    assert 'First' not in an_domain and 'Second' not in an_domain and 'Third' not in an_domain
    assert 'First' not in an_dist_bins and 'Second' not in an_dist_bins and 'Third' not in an_dist_bins


def test_errors_dataschema_anonymize(datasets):
    tips, titanic = datasets

    schma = DataSchema().generate(titanic)
    with pytest.raises(UserWarning) as w:
        schma.anonymize({})
        assert "To anonymise, it is necessary to use a dictionary with the format: {'var1':anonymizer1, 'var2':anonymizer2}" in str(w.value)

    an_encrypt = Anonymize(0)
    an_encrypt.set_key("07jaPY")
    with pytest.raises(ValueError) as e:
        schma.anonymize({'fare' : an_encrypt})
        assert "Input Error: Anonymize only supports Categorical or Binary variables ->" in str(e.value)

    with pytest.raises(ValueError) as e:
        schma.anonymize({'farer' : an_encrypt})
        assert "Input Error: Keys of 'anonymize_params' dictionary must be columns name of the data schema" in str(e.value)

    with pytest.raises(UserWarning) as w:
        schma.deanonymize({})
        assert "To De-anonymise, it is necessary to use a dictionary with the format: {'var1':anonym1, 'var2':anonym2}"

    with pytest.raises(ValueError) as e:
        schma.deanonymize({'fare' : an_encrypt})
        assert "Input Error: Deanonymize only supports Categorical or Binary variables ->" in str(e.value)

    with pytest.raises(ValueError) as e:
        schma.deanonymize({'farer' : an_encrypt})
        assert "Input Error: Deanonymize only supports Categorical or Binary variables ->" in str(e.value)

def test_dataschema_properties(datasets):
    tips, titanic = datasets

    schma = DataSchema().generate(titanic)
    assert ['pclass', 'sibsp', 'parch', 'embarked', 'class', 'who', 'deck', 'embark_town'] == schma.categorical_feats
    assert ['age', 'fare'] == schma.continuous_feats
    assert ['survived', 'sex', 'adult_male', 'alive', 'alone'] == schma.binary_feats
    assert len(schma.discrete_feats) == 0


def test_generate_manual(datasets):
    tips, titanic = datasets
    schma = DataSchema().generate_manual(
        titanic,
        categ_columns=['class'],
        discrete_columns=['age'],
        binary_columns=['survived', 'alive']
    )

    for key, item in schma.feats.items():
        if key == 'class':
            assert isinstance(item, CategoricalFeature)

        if key == 'age':
            assert isinstance(item, DiscreteFeature)

        if key == 'alive' or key == 'survived':
            assert isinstance(item, BinaryFeature)

        if key  not in ('class', 'age', 'alive', 'survived'):
            assert isinstance(item, ContinuousFeature)

    # assert everything is continuous by default
    schma = DataSchema().generate_manual(
        titanic,
        categ_columns=[],
        discrete_columns=[],
        binary_columns=[]
    )

    for _, item in schma.feats.items():
        assert isinstance(item, ContinuousFeature)


def test_validate(datasets):
    tips, titanic = datasets

    titanic2 = titanic.copy()
    titanic2['deck'] = 0

    schma = DataSchema().generate(titanic)
    schma2 = DataSchema().generate(titanic2)

    with pytest.raises(RuntimeError) as exinfo:
        schma.validate(schma2)

    assert "Data types types do not match. 'deck' in other is DataType.INTEGER. However, DataType.STRING is expected." in str(exinfo.value)

    titanic2 = titanic.drop('deck', axis=1)
    schma2 = DataSchema().generate(titanic2)

    with pytest.raises(RuntimeError) as exinfo:
        schma.validate(schma2)

    assert "Features do not match." in str(exinfo.value)


def test_serialization(datasets, tmpdir):
    tips, titanic = datasets

    schma = DataSchema().generate(titanic).calculate_statistics()
    path = str(tmpdir) + '/schema.json'
    schma.save(path)
    recovered = DataSchema.load(path)

    # If any of this fail, the serialization is wrong
    schma.validate(recovered)
    recovered.validate(schma)


def test_get_features_by_type(datasets):
    tips, titanic = datasets
    schema = DataSchema().generate(titanic)

    str_feats = {'class', 'alive', 'deck', 'embark_town', 'embarked', 'sex', 'who'}
    float_feats = {'age', 'fare'}
    assert set(schema.get_features_by_type(DataType.STRING)) == set(str_feats)
    assert set(schema.get_features_by_type(DataType.FLOAT)) == set(float_feats)


def test_subtypes():
    # Test added after bug discovery that float32 were not assigned to continuous

    df = pd.DataFrame(data={"float": np.random.uniform(size=1000)})
    df["float_64"] = df["float"].astype(np.float64)
    df["float_32"] = df["float"].astype(np.float32)
    df["float_16"] = df["float"].astype(np.float16)
    df["int_64"] = (df["float_64"] * 10000).astype(np.int64)
    df["int_32"] = (df["float_64"] * 10000).astype(np.int32)
    df["int_16"] = (df["float_64"] * 10000).astype(np.int16)
    df["uint_64"] = (df["float_64"] * 10000).astype(np.uint64)
    df["uint_32"] = (df["float_64"] * 10000).astype(np.uint32)
    df["uint_16"] = (df["float_64"] * 10000).astype(np.uint16)

    schema = DataSchema().generate(df)

    assert all(elem in schema.continuous_feats  for elem in ['float', 'float_64', 'float_32', 'float_16'])
    assert all(elem in schema.discrete_feats  for elem in ['int_64', 'int_32', 'int_16'])


    assert isinstance(schema.feats['float_64'], ContinuousFeature)
    assert isinstance(schema.feats['float_32'], ContinuousFeature)
    assert isinstance(schema.feats['float_16'], ContinuousFeature)
    assert isinstance(schema.feats['int_64'], DiscreteFeature)
    assert isinstance(schema.feats['int_32'], DiscreteFeature)
    assert isinstance(schema.feats['int_16'], DiscreteFeature)


    assert schema.feats['float_64'].dtype == DataType.FLOAT
    assert schema.feats['float_32'].dtype == DataType.FLOAT
    assert schema.feats['float_16'].dtype == DataType.FLOAT
    assert schema.feats['int_64'].dtype == DataType.INTEGER
    assert schema.feats['int_32'].dtype == DataType.INTEGER
    assert schema.feats['int_16'].dtype == DataType.INTEGER


def test_pandas_categorical_type():
    # Test added after bug discovery that schemas with dataframes with categorical type raise Exception

    df = pd.DataFrame(data={
        'categorical_int': np.random.choice([0,1,2,3], size=100),
        'categorical_str': np.random.choice(["A", "B", "C", "D"], size=100)
    })
    df["categorical_int"] = df["categorical_int"].astype("category")
    df["categorical_str"] = df["categorical_str"].astype("category")

    schema = DataSchema().generate(df)
    assert all(elem in schema.categorical_feats  for elem in ['categorical_int', 'categorical_str'])
    assert isinstance(schema.feats['categorical_int'], CategoricalFeature)
    assert isinstance(schema.feats['categorical_str'], CategoricalFeature)


def test_float_conversions():

    df = pd.DataFrame(data={
        'float_categorical': np.random.choice([0., 1., 2.], size=1000),
        'float_discrete': np.random.randint(0, 10000, size=1000).astype(float),
        'float_continous': np.random.uniform(0, 10000, size=1000)
    })
    schema = DataSchema().generate(df)
    assert isinstance(schema.feats['float_categorical'], CategoricalFeature)
    assert isinstance(schema.feats['float_discrete'], DiscreteFeature)
    assert isinstance(schema.feats['float_continous'], ContinuousFeature)


def test_categorical_and_numerical_user_assignation():

    # Test cat_feats and num_feats params to manually assign feature types to avoid automatic inference
    df = pd.DataFrame(data={
        'float_categorical': np.random.choice([0., 1., 2.], size=1000),
        'float_discrete': np.random.randint(0, 10000, size=1000).astype(float),
        'float_continous': np.random.uniform(0, 10000, size=1000),
        'int_categorical': np.random.choice([0, 1, 2], size=1000),
        'int_discrete': np.random.randint(0, 10000, size=1000)
    })

    # Generate initially with automatic inference
    schema = DataSchema().generate(df)
    assert isinstance(schema.feats['float_categorical'], CategoricalFeature)
    assert isinstance(schema.feats['float_discrete'], DiscreteFeature)
    assert isinstance(schema.feats['float_continous'], ContinuousFeature)
    assert isinstance(schema.feats['int_categorical'], CategoricalFeature)
    assert isinstance(schema.feats['int_discrete'], DiscreteFeature)

    # Generate now with manual assignation of data datatypes
    cat_feats = ['float_discrete', 'float_continous', 'int_discrete']
    num_feats = ['float_categorical', 'int_categorical']
    schema = DataSchema().generate(
        df,
        force_types=dict({c: FeatType.CATEGORICAL for c in cat_feats}, **{n: FeatType.DISCRETE for n in num_feats})
    )
    # assert isinstance(schema.feats['float_categorical'], DiscreteFeature)
    # assert isinstance(schema.feats['float_discrete'], CategoricalFeature)
    # assert isinstance(schema.feats['float_continous'], CategoricalFeature)
    # assert isinstance(schema.feats['int_categorical'], DiscreteFeature)
    # assert isinstance(schema.feats['int_discrete'], CategoricalFeature)
    #
    # # Specifying already most correct features types, keeps them
    # schema = DataSchema().generate(
    #     df,
    #     cat_feats=['float_categorical', 'int_categorical'],
    #     num_feats=['int_discrete', 'float_discrete', 'float_continous']
    # )
    # assert isinstance(schema.feats['float_categorical'], CategoricalFeature)
    # assert isinstance(schema.feats['float_discrete'], DiscreteFeature)
    # assert isinstance(schema.feats['float_continous'], ContinuousFeature)
    # assert isinstance(schema.feats['int_categorical'], CategoricalFeature)
    # assert isinstance(schema.feats['int_discrete'], DiscreteFeature)

    # If a feature is specified both as numerical and categorical, then an exception is raised
    # with pytest.raises(ValueError) as exinfo:
    #     schema = DataSchema().generate(df, cat_feats=['float_discrete'], num_feats=['float_discrete'])

    # String column as a numeric doesn't change it (raises warning)
    df['str_float_categorical'] = df['float_categorical'].astype(str)
    schema = DataSchema().generate(df)
    assert isinstance(schema.feats['str_float_categorical'], CategoricalFeature)

