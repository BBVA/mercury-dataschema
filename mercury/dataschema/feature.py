from enum import Enum

import numpy as np
import warnings


class DataType(Enum):
    INTEGER = 'INTEGER'
    FLOAT = 'FLOAT'
    STRING = 'STRING'
    DATE = 'DATE'
    BOOL = 'BOOL'
    CATEGORICAL = 'CATEGORICAL'  # for pandas categorical type
    UNKNOWN = 'UNKNOWN'


class FeatType(Enum):
    BINARY = 'BINARY'
    CATEGORICAL = 'CATEGORICAL'
    DISCRETE = 'DISCRETE'
    CONTINUOUS = 'CONTINUOUS'
    UNKNOWN = 'UNKNOWN'


class Feature:
    """ This class represents a generic feature within a schema.

    Args:
        name: Feature name
        dtype: Data type of the feature
    """
    def __init__(self,
                 name: str = None,
                 dtype: DataType = None
                 ):
        self.name = name
        self.dtype = dtype if dtype else DataType.UNKNOWN
        self.stats = {}
        self.cache = {}  # Intermediate heavy calculations

    def build_stats(self, column, calculator=None):
        no_nan_col = column.dropna()
        uniques = no_nan_col.unique()
        self.stats['num_nan'] = int(column.isna().sum())
        self.stats['percent_nan'] = float(self.stats['num_nan'] / len(column))
        self.stats['samples'] = len(column)
        self.stats['percent_unique'] = float(len(uniques) / self.stats['samples'])
        self.stats['cardinality'] = len(uniques)
        self.cache['uniques'] = uniques.tolist()

        self.cache['no_nan_filtered'] = no_nan_col  # TODO: This could be inefficient

        return self

    def __str__(self):
        return f"Feature (NAME={self.name}, dtype={self.dtype})"

    def __repr__(self):
        return self.__str__()

    def _get_enum_feat_type(self):
        return FeatType.UNKNOWN

    @property
    def as_enum(self):
        return self._get_enum_feat_type()

    def to_json(self) -> dict:
        stats_serialized = self.stats.copy()

        for key, val in stats_serialized.items():
            if isinstance(val, int):
                stats_serialized[key] = int(val)
            if isinstance(val, bool):
                stats_serialized[key] = bool(val)
            if isinstance(val, float):
                stats_serialized[key] = float(val)

        return {'name': self.name, 'dtype': self.dtype.value, 'stats': stats_serialized, 'feat_type': self.as_enum.value}


class BinaryFeature(Feature):
    """ This class represents a binary feature within a schema
    (i.e. only two possible values).

    Args:
        name: Feature name
        dtype: Data type of the feature
    """
    def __init__(self,
                 name=None,
                 dtype=None
                 ):

        super().__init__(name, dtype)

    def build_stats(self, column, calculator):
        super().build_stats(column, calculator)
        self.stats['domain'] = self.cache['uniques']
        calculator.distribution(column, self, bins=2)
        return self

    def __str__(self):
        return f"Binary Feature (NAME={self.name}, dtype={self.dtype})"

    def __repr__(self):
        return self.__str__()

    def _get_enum_feat_type(self):
        return FeatType.BINARY


class CategoricalFeature(Feature):
    """ This class represents a categorical feature within a schema
    (i.e. only N possible values).

    Args:
        name: Feature name
        dtype: Data type of the feature
    """
    def __init__(self,
                 name=None,
                 dtype=None
                 ):

        super().__init__(name, dtype)

    def build_stats(self, column, calculator):
        super().build_stats(column, calculator)
        limit = calculator.limit_categorical_perc

        if isinstance(limit, (int, float)):

            if limit <= 0 or limit >= 1:
                raise ValueError("Input Error: 'limit_categorical_perc' must be a float between 0 and 1")

            elif len(self.cache['uniques']) / self.stats['samples'] > limit:
                warnings.warn(f"{self.name} will be truncated in both statistics 'domain' and 'distribution' with the most frequent values")
                #  We get the N most frequent values according to the dataset size
                self.stats['domain'] = list(column.value_counts().index[:int(limit * self.stats['samples'])])

            else:  # Low cardinality
                self.stats['domain'] = self.cache['uniques']

        else:
            self.stats['domain'] = self.cache['uniques']

        calculator.distribution(column, self)

        return self

    def __str__(self):
        return f"Categorical Feature (NAME={self.name}, dtype={self.dtype})"

    def __repr__(self):
        return self.__str__()

    def _get_enum_feat_type(self):
        return FeatType.CATEGORICAL


class DiscreteFeature(Feature):
    """ This class represents a discrete feature within a schema
    (i.e. any number without decimals).

    Args:
        name: Feature name
        dtype: Data type of the feature
    """
    def __init__(self,
                 name=None,
                 dtype=None
                 ):

        super().__init__(name, dtype)

    def build_stats(self, column, calculator):
        super().build_stats(column, calculator)
        calculator.min(column, self)
        calculator.max(column, self)
        calculator.distribution(column, self)
        return self

    def __str__(self):
        return f"Discrete Feature (NAME={self.name}, dtype={self.dtype})"

    def __repr__(self):
        return self.__str__()

    def _get_enum_feat_type(self):
        return FeatType.DISCRETE


class ContinuousFeature(Feature):
    """ This class represents a continuous feature within a schema
    (e.g. a float).

    Args:
        name: Feature name
        dtype: Data type of the feature
    """
    def __init__(self,
                 name=None,
                 dtype=None
                 ):

        super().__init__(name, dtype)

    def build_stats(self, column, calculator):
        super().build_stats(column, calculator)
        calculator.min(column, self)
        calculator.max(column, self)
        calculator.mean(column, self)
        calculator.std(column, self)
        calculator.distribution(column, self)
        return self

    def __str__(self):
        return f"Continuous Feature (NAME={self.name}, dtype={self.dtype})"

    def __repr__(self):
        return self.__str__()

    def _get_enum_feat_type(self):
        return FeatType.CONTINUOUS


class FeatureFactory:

    def __init__(self):
        pass

    def infer_datatype(self, column: "pandas.Series", feature: Feature) -> DataType:  # noqa: F821
        """ Finds out the data type of the column.

        Args:
            column: column which datatype will be inferred
            feature: Feature object. This is needed because we want to cache several internal
                     operations, so future calls are faster.

        Returns:
            Returns the datatype of the column
        """
        datatype = DataType.UNKNOWN

        if column.dtype.name == 'category':
            datatype = DataType.CATEGORICAL
        elif np.issubdtype(column, np.integer):
            datatype = DataType.INTEGER
        elif np.issubdtype(column, np.bool_):
            datatype = DataType.BOOL
        elif np.issubdtype(column, np.floating):
            datatype = DataType.FLOAT
        elif np.issubdtype(column, np.object_):
            sample = feature.cache['no_nan_filtered'].iloc[0]
            if type(sample) is str:
                datatype = DataType.STRING
            # TODO: Este tipo puede ser otro array
            # TODO: Este tipo puede ser un json (dict)
            # TODO: Este tipo puede ser un datetime

        return datatype

    def _build_dummy_feature(self, datatype: DataType, feat_type: FeatType, name: str) -> Feature:
        """ Returns a dummy and uninitialized feature. This method is not intended to be
        used apart from serialization purposes.
        """
        feat = Feature()
        if feat_type == FeatType.BINARY:
            feat = BinaryFeature()
        if feat_type == FeatType.CATEGORICAL:
            feat = CategoricalFeature()
        if feat_type == FeatType.DISCRETE:
            feat = DiscreteFeature()
        if feat_type == FeatType.CONTINUOUS:
            feat = ContinuousFeature()
        feat.dtype = datatype
        feat.name = name

        return feat

    def _infer_feature_type_from_float(self, feat, threshold_categorical, colname, verbose=False):
        if (feat.cache['no_nan_filtered'] % 1 == 0).all():  # The float column doesn't contain decimals
            if (feat.stats['percent_unique'] < threshold_categorical):
                # Case Categorical as float
                if verbose:
                    warnings.warn(
                        f"""FLOAT feature {colname} converted to Categorical because percentage of unique """
                        f"""values {feat.stats['percent_unique']} is lower than threshold {threshold_categorical}""",
                        RuntimeWarning
                    )
                return FeatType.CATEGORICAL

            # Case Discrete as Float
            return FeatType.DISCRETE

        # If it does contain decimals, directly create Continuous Feature (categorical feature would rarely be
        # codified as floats with decimals
        return FeatType.CONTINUOUS

    def _infer_feature_type_from_int(self, feat, colname, threshold_categorical, verbose=False):
        if feat.stats['percent_unique'] >= threshold_categorical:
            return FeatType.DISCRETE
        else:
            if verbose:
                warnings.warn(
                    f"""INTEGER feature {colname} converted to Categorical because percentage of unique """
                    f"""values {feat.stats['percent_unique']} is lower than threshold {threshold_categorical}""",
                    RuntimeWarning
                )
            return FeatType.CATEGORICAL

    def build_feature(self,
                      column: 'pandas.Series',  # noqa: F821
                      colname: str = None,
                      threshold_categorical: float = 1e-5,
                      force_feat_type: FeatType = None,
                      verbose: bool = True
                      ) -> Feature:
        """ Builds a schema Feature object given a column.

        Args:
            column: Column to be analyzed
            colname: Name of the column (feature)
            threshold_categorical: percentage of necessary unique values for a feature to be considered
                           categorical. If the percentage of unique values < cat_threshold, the
                           column will be taken as categorical. This parameter can be a single float
                           (same threshold for all columns) or a dict in which each key is the name of
                           the column. Use the later for custom thresholds per column.
            force_feat_type: If user wants to force a variable to be of certain type, he/she can use
                            this parameter and its type will not be auto-inferred, but set to this.
            verbose: If this is set to False, possible inner warnings won't be shown.

        Returns:
            Feature with only the base statistics calculated
        """
        feat = Feature().build_stats(column)
        datatype = self.infer_datatype(column, feat)
        feat_type = FeatType.UNKNOWN

        # If user forces the feature type we kindly fulfill his/her wishes
        if force_feat_type is not None:
            feat = self._build_dummy_feature(datatype, force_feat_type, colname)
            feat.stats.update(feat.stats)
            return feat

        if feat.stats['cardinality'] == 2:
            feat_type = FeatType.BINARY
        else:
            # Data could still be either categorical, discrete or continuous
            if datatype is DataType.FLOAT:
                feat_type = self._infer_feature_type_from_float(feat, threshold_categorical, colname, verbose=verbose)

            if datatype is DataType.INTEGER:
                feat_type = self._infer_feature_type_from_int(feat, colname, threshold_categorical, verbose=verbose)

            if (datatype is DataType.STRING) or (datatype is DataType.CATEGORICAL):
                feat_type = FeatType.CATEGORICAL

        featret = self._build_dummy_feature(datatype, feat_type, colname)
        featret.stats.update(feat.stats)
        return featret
