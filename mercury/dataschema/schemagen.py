import json
import numpy as np

from typing import Union, List, Dict

from .feature import (
    FeatureFactory,
    ContinuousFeature,
    DiscreteFeature,
    CategoricalFeature,
    BinaryFeature
)

from .feature import (
    DataType,
    FeatType
)

from .calculator import StatCalculatorFactory


class DataSchema:
    """ Dataset schema

    This class takes a dataframe and generates its schema as a collection of feature.
    Feature objects. Each one of them will contain metadata and statistics about a
    column of the original dataframe that can be further exploded.


    Example:
        ```python
        >>> schma = DataSchema()\
        >>>            .generate(dataset)\
        >>>            .calculate_statistics()
         'DISBURSED_AMOUNT': Categorical Feature (NAME=DISBURSED_AMOUNT, dtype=DataType.INTEGER),
         'ASSET_COST': Categorical Feature (NAME=ASSET_COST, dtype=DataType.INTEGER),
         'LTV': Continuous Feature (NAME=LTV, dtype=DataType.FLOAT),
         'BUREAU_SCORE': Discrete Feature (NAME=BUREAU_SCORE, dtype=DataType.INTEGER),
         'BUREAU_SCORE_DESCRIPTION': Categorical Feature (NAME=BUREAU_SCORE_DESCRIPTION, dtype=DataType.STRING),
         'NEW_LOANS_IN_LAST_SIX_MONTHS': Discrete Feature (NAME=NEW_LOANS_IN_LAST_SIX_MONTHS, dtype=DataType.INTEGER),
         'DEFAULTED_LOANS_IN_LAST_SIX_MONTHS': Discrete Feature (NAME=DEFAULTED_LOANS_IN_LAST_SIX_MONTHS, dtype=DataType.INTEGER),
         'NUM_LOANS_TAKEN': Discrete Feature (NAME=NUM_LOANS_TAKEN, dtype=DataType.INTEGER),
         'NUM_ACTIVE_LOANS': Discrete Feature (NAME=NUM_ACTIVE_LOANS, dtype=DataType.INTEGER),
         'NUM_DEFAULTED_LOANS': Discrete Feature (NAME=NUM_DEFAULTED_LOANS, dtype=DataType.INTEGER),
         'AGE': Discrete Feature (NAME=AGE, dtype=DataType.INTEGER),
         'GENDER': Binary Feature (NAME=GENDER, dtype=DataType.STRING),
         'CIVIL_STATUS': Categorical Feature (NAME=CIVIL_STATUS, dtype=DataType.STRING),
         'ORIGIN': Binary Feature (NAME=ORIGIN, dtype=DataType.STRING),
         'DIGITAL': Binary Feature (NAME=DIGITAL, dtype=DataType.INTEGER),
         'SCORE': Continuous Feature (NAME=SCORE, dtype=DataType.FLOAT),
         'PREDICTION': Binary Feature (NAME=PREDICTION, dtype=DataType.INTEGER)}
        >>> schma.feats['SCORE'].stats
        {'num_nan': 0,
        'percent_nan': 0.0,
        'samples': 233154,
        'percent_unique': 0.7967352050576014,
        'cardinality': 185762,
        'min': 0.17454321487679067,
        'max': 0.9373813084029072,
        'mean': 0.7625553210045813,
        'std': 0.15401509786623635,
        'distribution': array([7.48617716e-07, 1.07579979e-06, 1.40298186e-06, 1.73016394e-06,
                2.05734601e-06, 2.38452809e-06, 2.71171016e-06, 3.03889224e-06,
                3.36607431e-06, 3.69325638e-06, 4.02043846e-06])}
        # Specifying custom parameters (shared among all features) for the calculate_statistics method
        >>> schma = DataSchema()\
        ...    .generate(dataset)\
        ...    .calculate_statistics({'distribution_bins_method': 'sqrt'})  # Specify bin generation method (see numpy.hist)

        # We can also specify granular statistic parameters per variable
        >>> schma = DataSchema()\
        ...    .generate(dataset)\
        ...    .calculate_statistics({'SCORE': {'distribution_bins_method': 'sqrt'}})  # Specify bin generation method (see numpy.hist)

        >>> schma = DataSchema()\
        ...    .generate(dataset)\
        ...    .calculate_statistics({'SCORE': {'distribution_bins_method': 5}})  # Specify 5 bins only for numerical features
        ```
    """
    def __init__(self):
        self.dataframe = None
        self.feats = {}
        self._feat_factory = None
        self._generated = False

    def generate_manual(
        self,
        dataframe: Union["pandas.DataFrame", "pyspark.sql.DataFrame"],  # noqa: F821
        categ_columns: List[str],
        discrete_columns: List[str],
        binary_columns: List[str],
        custom_stats: dict = None,
    ) -> "DataSchema":
        """ Builds the schema manually. This acts like `generate()` but in a more restrictive way.
        All the names passed to `categ_columns` will be taken as categorical features, no more, no less.
        It will avoid making automatic type inference on every feature not in `categ_columns`.
        The same rule is applied on `discrete_columns`.

        Note:
            This method is considered to be low level. If you use this, make sure the type assignment
            to each feature type is compatible with the datatypes (float, int, string,...) in the column or
            a later call to `calculate_statistics` could fail.

        Args:
            dataframe: DataFrame on which the schema will be inferred.
            categ_columns: list of columns which will be forced to be taken as categorical. Warning:
                          all features not in this list are guaranteed not being categorical
            discrete_columns: list of columns which will be forced to be taken as discrete. Warning:
                          all features not in this list are guaranteed not to be taken as discrete (i.e.
                          they will be continuous).
            binary_columns: list of column which will be forced to be taken as binary.
            custom_stats: Custom statistics to be calculated for each column.
            verbose: whether to show or filter all possible warning messages
        """
        force_types = {}
        for col in dataframe.columns:
            if col in categ_columns:
                force_types[col] = FeatType.CATEGORICAL
            else:
                # Is in either binary, continuous or discrete lists
                if col in discrete_columns:
                    force_types[col] = FeatType.DISCRETE
                elif col in binary_columns:
                    force_types[col] = FeatType.BINARY
                else:
                    force_types[col] = FeatType.CONTINUOUS

        return self.generate(
            dataframe=dataframe,
            force_types=force_types,
            verbose=False,
            custom_stats=custom_stats
        )

    def generate(
        self,
        dataframe: Union["pandas.DataFrame", "pyspark.sql.DataFrame"],  # noqa: F821
        force_types: Dict[str, FeatType] = None,
        custom_stats: dict = None,
        verbose: bool = True,
    ) -> "DataSchema":
        """ Builds the schema. For float and integer datatypes, by default the method tries to infer
            if a feature is categorical or numeric (Continuous or Discrete) depending on the percentage
            of unique values. However, that doesn't work in all the cases. In those cases, you can use
            the `force_types` param to specify which features should be categorical and which
            should be numeric independently of the percentage of unique values.

        Args:
            dataframe: DataFrame on which the schema will be inferred.
            force_types: Dictionary with the form <FEATURE_NAME, FeatType> that contains the features to be
                        forced to a specific type (Continuous, Discrete, Categorical...)
            custom_stats: Custom statistics to be calculated for each column
            verbose: whether to show or filter all possible warning messages
        """
        if "pyspark" in str(type(dataframe)):
            raise RuntimeError("Sorry, Pyspark is not supported yet...")

        self.dataframe = dataframe
        self._generated = True

        self._feat_factory = FeatureFactory()

        inferring_types = True if force_types is None else False

        for col in self.dataframe.columns:
            thresh = self._get_threshold(len(self.dataframe))

            # Look if the feature type has been specified
            forced_type = None
            if not inferring_types and col in force_types:
                forced_type = force_types[col]

            feat = self._feat_factory.build_feature(
                self.dataframe.loc[:, col],
                col,
                force_feat_type=forced_type,
                threshold_categorical=thresh,
                verbose=inferring_types and verbose  # Only show warnings (if any) when using default args.
            )
            self.feats[col] = feat

        return self

    def anonymize(self, anonymize_params: dict) -> "DataSchema":
        """
        Anonymize the selected features of a data schema.

        Args:
            anonymize_params: Dictionary where the keys are the names of the columns to be anonymized and the values
                              are mercury.contrib.dataschema.Anonymize objects that can be used to anonymize them.
        Raises:
            UserWarning, if anonymize_params is empty.
            ValueError, if the feature selected to deanonymize is not binary or categorical, or is not a feature of the dataschema.
        """
        if not anonymize_params:
            raise UserWarning("To anonymize, it is necessary to use a dictionary with the format: {'var1':anonymizer1, 'var2':anonymizer2}")

        if any(feat not in self.feats.keys() for feat in anonymize_params.keys()):
            raise ValueError("Input Error: Keys of 'anonymize_params' dictionary must be columns name of the data schema")

        for feature in list(self.feats.keys()):
            anon = anonymize_params.get(feature)

            if anon:
                if not isinstance(self.feats[feature], (BinaryFeature, CategoricalFeature)):
                    raise ValueError(f"Input Error: Anonymze only supports Categorical or Binary variables -> {feature}, You can use \
                                        the `force_types` param in 'generate()' to specify which features should be categorical ")
                else:
                    self.feats[feature].stats['distribution_bins'] = anon.\
                        anonymize_list_any_type(list(self.feats[feature].stats['distribution_bins']))
                    self.feats[feature].stats['domain'] = anon.\
                        anonymize_list_any_type(list(self.feats[feature].stats['domain']))

        return self

    def deanonymize(self, anonymize_params: dict) -> "DataSchema":
        """
        De-anonymize the selected features on a preloaded schema.

        Args:
            anonymize_params: Dictionary where the keys are the names of the columns to be deanonymized and the values
                              are mercury.contrib.dataschema.Anonymize objects that can be used to deanonymize them.

        Raises:
            UserWarning, if anonymize_params is empty.
            ValueError, if the feature selected to deanonymize is not binary or categorical, or is not a feature of the dataschema.
        """
        if not anonymize_params:
            raise UserWarning("To De-anonymize, it is necessary to use a dictionary with the format: {'var1':anonym1, 'var2':anonym2}")

        if any(feat not in self.feats.keys() for feat in anonymize_params.keys()):
            raise ValueError("Input Error: Keys of 'anonymize_params' dictionary must be columns name of the data schema")

        for feature in list(self.feats.keys()):
            anon = anonymize_params.get(feature)

            if anon:

                if not isinstance(self.feats[feature], (BinaryFeature, CategoricalFeature)):
                    raise ValueError(f"Input Error: Deanonymize only supports Categorical or Binary variables -> {feature} ")
                else:
                    operation = int if self.feats[feature].dtype == DataType.INTEGER else str
                    self.feats[feature].stats['distribution_bins'] = \
                        list(map(operation, anon.deanonymize_list(self.feats[feature].stats['distribution_bins'])))
                    self.feats[feature].stats['domain'] = \
                        list(map(operation, anon.deanonymize_list(self.feats[feature].stats['domain'])))
        return self

    def calculate_statistics(
        self,
        calculator_configs: dict = None
    ) -> "DataSchema":
        """ Triggers the computation of all statistics for all registered features
        of the schema.

        Args:
            calculator_configs: Optional configurations for each of the calculator parameters.
                                This can be either a dict or a "dict of dicts". In the first case,
                                the statistics for ALL FEATURES will be computed with those parameters.
                                Additionally, you can specify a mapping of [feature_name: {config}] with
                                granular configurations per feature.
                                The supported configuration keys are the attributes declared within a calculator class.
                                See mercury.contrib.dataschema.calculator.PandasStatCalculator (or Spark) for details.
        """
        featnames = list(self.feats.keys())

        calculator_configs = calculator_configs if calculator_configs else {}

        # User can pass us two  types:
        #  - {'param': 'value', 'param2': 'value'} -> Single config shared for all variables
        #  - {{config_var1}, {config_var2}, {config_var3}, ...} -> 1 config per variable
        multiple_configs = len(calculator_configs) > 0 and isinstance(list(calculator_configs.values())[0], dict)

        # Case when user pass a single shared config
        if not multiple_configs:
            calculator = StatCalculatorFactory.build_calculator(self.dataframe)
            calculator.set_config(**calculator_configs)

        for feature in featnames:
            if multiple_configs:
                # Case when user pass one config per variable
                calculator = StatCalculatorFactory.build_calculator(self.dataframe)
                if feature in calculator_configs:
                    calculator.set_config(**(calculator_configs[feature]))

            # Calculate distributions
            self.feats[feature].build_stats(self.dataframe.loc[:, feature], calculator)

        return self

    def _get_threshold(self, dataset_size):
        """ Calculates a dynamic threshold for determining whether a variable is categorical
        given the dataset. It uses an asymptotic function (whose lim->0) clipped to a maximum value of 1.
        """
        return np.minimum(1, 50 / (dataset_size))

    @property
    def continuous_feats(self) -> List[str]:
        """ List with the names of all continuous features
        """
        return [key for key, feat in self.feats.items() if isinstance(feat, ContinuousFeature)]

    @property
    def categorical_feats(self) -> List[str]:
        """ List with the names of all categorical features
        """
        return [key for key, feat in self.feats.items() if isinstance(feat, CategoricalFeature)]

    @property
    def binary_feats(self) -> List[str]:
        """ List with the names of all binary features
        """
        return [key for key, feat in self.feats.items() if isinstance(feat, BinaryFeature)]

    @property
    def discrete_feats(self) -> List[str]:
        """ List with the names of all discrete features
        """
        return [key for key, feat in self.feats.items() if isinstance(feat, DiscreteFeature)]

    def validate(self, other: "DataSchema"):
        """ Validates other schema with this one. The other schema will be considered
        valid if it shares the same feature names and datatypes with this.

        Args:
            other: other schema to be checked from this one

        Raises:
            RuntimeError if other schema differs from this one
        """
        # Check feature names match
        if list(self.feats.keys()) != list(other.feats.keys()):
            diff = set(self.feats.keys()) - set(other.feats.keys())
            raise RuntimeError(f"Features do not match. These ones are not present on both datasets {list(diff)}")

        # Check feature and data types are the same
        for key, item in other.feats.items():
            if not isinstance(item, self.feats[key].__class__):
                raise RuntimeError(f"""Feature types do not match. '{key}' in other is """
                                   f"""{type(item)}. However, {type(self.feats[key])} is expected.""")

            if item.dtype != self.feats[key].dtype:
                raise RuntimeError(f"""Data types types do not match. '{key}' in other is """
                                   f"""{item.dtype}. However, {self.feats[key].dtype} is expected.""")

    def to_json(self) -> dict:
        """ Converts the schema to a JSON representation

        Returns:
            dictionary with the features and their stats
        """
        retdict = dict(feats=dict())
        for key, val in self.feats.items():
            retdict['feats'][key] = self.feats[key].to_json()

        return retdict

    def save(self, path):
        """ Saves a JSON with the schema representation

        Args:
            path: where the JSON will be saved.
        """
        with open(path, 'w') as file:
            json.dump(self.to_json(), file)

    @classmethod
    def load(cls, path: str) -> "DataSchema":
        """ Loads a previously serialized schema (as JSON)

        Args:
            path: path to the serialized schema

        Returns:
            The rebuilt schema
        """
        with open(path, 'r') as file:
            json_obj = json.load(file)
        schema = cls.from_json(json_obj)
        return schema

    @classmethod
    def from_json(cls, json_obj: dict) -> "DataSchema":
        """ Rebuilds an schema from a JSON representation.

        Returns:
            The rebuild schema
        """
        schema = DataSchema()
        factory = FeatureFactory()

        for featname, feat in json_obj['feats'].items():
            ftype = FeatType[feat['feat_type']]
            dtype = DataType[feat['dtype']]
            feat_name = feat['name']
            dummy_feat = factory._build_dummy_feature(dtype, ftype, feat_name)
            dummy_feat.stats = feat['stats']
            schema.feats[featname] = dummy_feat

        return schema

    def get_features_by_type(self, datatype: DataType):
        return [key for key, feat in self.feats.items() if feat.dtype == datatype]
