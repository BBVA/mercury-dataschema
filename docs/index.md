# mercury-dataschema

[![](https://github.com/BBVA/mercury-dataschema/actions/workflows/test.yml/badge.svg)](https://github.com/BBVA/mercury-dataschema)
![](https://img.shields.io/badge/latest-0.0.1-blue)

`mercury-dataschema` is a submodule of the Mercury library which acts as a utility tool that, given a Pandas DataFrame, its `DataSchema` class auto-infers feature types and automatically calculates different statistics depending on them.

This type inference isn't solely based on data types but in the information the variables contain. For example: if a feature is encoded as a `float` but its cardinality is 2, we can be sure it's a binary feature.

This package is used by other Mercury submodules, and you also can use it separately from the rest of the library. 

As an idea (there are plenty of them, though), it is particularly useful when preprocessing datasets. Having to specify the typical `categorical_cols` and `coninuous_cols` is over!

## Mercury project at BBVA

Mercury is a collaborative library that was developed by the Advanced Analytics community at BBVA. Originally, it was created as an [InnerSource](https://en.wikipedia.org/wiki/Inner_source) project but after some time, we decided to release certain parts of the project as Open Source.
That's the case with the `mercury-dataschema` package. 

If you're interested in learning more about the Mercury project, we recommend reading this blog [post](https://www.bbvaaifactory.com/mercury-acelerando-la-reutilizacion-en-ciencia-de-datos-dentro-de-bbva/) from www.bbvaaifactory.com

## User installation

The easiest way to install `mercury-dataschema` is using ``pip``:

    pip install -U mercury-dataschema

## Example

```python
from mercury.dataschema.schemagen import DataSchema
from mercury.dataschema.feature import FeatType

dataset = UCIDataset().load()   # Any Dataframe 

schma = (DataSchema()         # Generate a lazy Schema object
    .generate(dataset)        # Manually trigger its construction (it mostly infers data types...)
    .calculate_statistics())  # Manually trigger extra statistic calculations for each feature
```

Then, we can inspect all the features with

```python
schma.feats
```

```
{'ID': Discrete Feature (NAME=None, dtype=DataType.INTEGER),
 'LIMIT_BAL': Discrete Feature (NAME=None, dtype=DataType.FLOAT),
 'SEX': Binary Feature (NAME=None, dtype=DataType.INTEGER),
 'EDUCATION': Categorical Feature (NAME=None, dtype=DataType.INTEGER),
 'MARRIAGE': Categorical Feature (NAME=None, dtype=DataType.INTEGER),
 'AGE': Discrete Feature (NAME=None, dtype=DataType.INTEGER),
 'PAY_0': Categorical Feature (NAME=None, dtype=DataType.INTEGER),
 'PAY_2': Categorical Feature (NAME=None, dtype=DataType.INTEGER),
 'PAY_3': Categorical Feature (NAME=None, dtype=DataType.INTEGER),
 'PAY_4': Categorical Feature (NAME=None, dtype=DataType.INTEGER),
 'PAY_5': Categorical Feature (NAME=None, dtype=DataType.INTEGER),
 'PAY_6': Categorical Feature (NAME=None, dtype=DataType.INTEGER),
 'BILL_AMT1': Discrete Feature (NAME=None, dtype=DataType.FLOAT),
 'BILL_AMT2': Discrete Feature (NAME=None, dtype=DataType.FLOAT),
 'BILL_AMT3': Discrete Feature (NAME=None, dtype=DataType.FLOAT),
 'BILL_AMT4': Discrete Feature (NAME=None, dtype=DataType.FLOAT),
 'BILL_AMT5': Discrete Feature (NAME=None, dtype=DataType.FLOAT),
 'BILL_AMT6': Discrete Feature (NAME=None, dtype=DataType.FLOAT),
 'PAY_AMT1': Discrete Feature (NAME=None, dtype=DataType.FLOAT),
 'PAY_AMT2': Discrete Feature (NAME=None, dtype=DataType.FLOAT),
 'PAY_AMT3': Discrete Feature (NAME=None, dtype=DataType.FLOAT),
 'PAY_AMT4': Discrete Feature (NAME=None, dtype=DataType.FLOAT),
 'PAY_AMT5': Discrete Feature (NAME=None, dtype=DataType.FLOAT),
 'PAY_AMT6': Discrete Feature (NAME=None, dtype=DataType.FLOAT),
 'default.payment.next.month': Binary Feature (NAME=None, dtype=DataType.INTEGER)}
```

And we can get extra feature statistics by inspecting the .stats attribute of the `Feature` objects.

```python
schma.feats['BILL_AMT4'].stats
```

```
{'num_nan': 0,
 'percent_nan': 0.0,
 'samples': 30000,
 'percent_unique': 0.7182666666666667,
 'cardinality': 21548,
 'min': -170000.0,
 'max': 891586.0,
 'distribution': [3.3333333333333335e-05,
  0.0,
  3.3333333333333335e-05,
  0.0,
  0.0,
  3.3333333333333335e-05,
  0.0,
  3.3333333333333335e-05,
  3.3333333333333335e-05,
  0.0,
  3.3333333333333335e-05,
  6.666666666666667e-05,
  6.666666666666667e-05,
  0.00016666666666666666,
  ...,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  3.3333333333333335e-05],
 'distribution_bins': [-170000.0,
  -163898.93103448275,
  -157797.8620689655,
  -151696.7931034483,
  ...,
  867181.724137931,
  873282.7931034482,
  879383.8620689653,
  885484.9310344828,
  891586.0]}
```

```python
schma.feats
```

```
{'ID': Discrete Feature (NAME=None, dtype=DataType.INTEGER),
 'LIMIT_BAL': Discrete Feature (NAME=None, dtype=DataType.FLOAT),
 'SEX': Binary Feature (NAME=None, dtype=DataType.INTEGER),
 'EDUCATION': Categorical Feature (NAME=None, dtype=DataType.INTEGER),
 'MARRIAGE': Categorical Feature (NAME=None, dtype=DataType.INTEGER),
 'AGE': Discrete Feature (NAME=None, dtype=DataType.INTEGER),
 'PAY_0': Categorical Feature (NAME=None, dtype=DataType.INTEGER),
 'PAY_2': Categorical Feature (NAME=None, dtype=DataType.INTEGER),
 'PAY_3': Categorical Feature (NAME=None, dtype=DataType.INTEGER),
 'PAY_4': Categorical Feature (NAME=None, dtype=DataType.INTEGER),
 'PAY_5': Categorical Feature (NAME=None, dtype=DataType.INTEGER),
 'PAY_6': Categorical Feature (NAME=None, dtype=DataType.INTEGER),
 'BILL_AMT1': Discrete Feature (NAME=None, dtype=DataType.FLOAT),
 'BILL_AMT2': Discrete Feature (NAME=None, dtype=DataType.FLOAT),
 'BILL_AMT3': Discrete Feature (NAME=None, dtype=DataType.FLOAT),
 'BILL_AMT4': Discrete Feature (NAME=None, dtype=DataType.FLOAT),
 'BILL_AMT5': Discrete Feature (NAME=None, dtype=DataType.FLOAT),
 'BILL_AMT6': Discrete Feature (NAME=None, dtype=DataType.FLOAT),
 'PAY_AMT1': Discrete Feature (NAME=None, dtype=DataType.FLOAT),
 'PAY_AMT2': Discrete Feature (NAME=None, dtype=DataType.FLOAT),
 'PAY_AMT3': Discrete Feature (NAME=None, dtype=DataType.FLOAT),
 'PAY_AMT4': Discrete Feature (NAME=None, dtype=DataType.FLOAT),
 'PAY_AMT5': Discrete Feature (NAME=None, dtype=DataType.FLOAT),
 'PAY_AMT6': Discrete Feature (NAME=None, dtype=DataType.FLOAT),
 'default.payment.next.month': Binary Feature (NAME=None, dtype=DataType.INTEGER)}
```

Note how for different features, the computed statistics vary:

```python
schma.feats['default.payment.next.month'].stats
```

```
{'num_nan': 0,
 'percent_nan': 0.0,
 'samples': 30000,
 'percent_unique': 6.666666666666667e-05,
 'cardinality': 2,
 'distribution': [0.7788, 0.2212],
 'distribution_bins': [0, 1],
 'domain': [1, 0]}
```

## Saving and loading schemas

You can serialize and reload `DataSchema`s so you can reuse them in the future.

```python
PATH = 'schma.json'
# Save the schema
schma.save(PATH)

# Load it back!
recovered = DataSchema.load(PATH)
```

## Help and support 

This library is currently maintained by a dedicated team of data scientists and machine learning engineers from BBVA AI Factory. 

### Documentation
website: https://bbva.github.io/mercury-dataschema/

### Email 
mercury.group@bbva.com
