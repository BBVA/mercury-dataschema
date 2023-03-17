import numpy as np
from typing import Union

from .feature import BinaryFeature, CategoricalFeature


class FeatureCalculator():
    """ This is a base class with the operation definitions. Several classes must
    extend this, implementing its operations for each one of the supported frameworks
    (namely Pandas and Pyspark)
    """

    def __init(self):
        pass

    def min(self, column, feature):
        pass

    def max(self, column, feature):
        pass

    def distribution(self, column, feature, bins=None):
        pass

    @property
    def _registered_params(self):
        return list(self.__dict__.keys())

    def set_config(self, **kwargs):
        """ Set attributes with the keys of the dictionary. These can be later used within
        specific calculator methods (like `distribution()` for specifying the number of bins).

        For this to work, the parameter must have been explicitly declared during object's
        constructor. That is, you cannot pass here a parameter name which the calculator doesn't
        support (or this will raise a ValueError).

        Args:
            **kwargs: The names and values of the desired parameters to set.

        Raises:
            ValueError if any keyword argument does not exist among the existing attributes of
            the object.
        """
        if kwargs is None:
            return

        for key, val in kwargs.items():
            if not hasattr(self, key):
                raise ValueError(
                    f"Error. This calculator doesn't support the `{key}` parameter. Available options are {self._registered_params}"
                )
            setattr(self, key, val)


class PandasStatCalculator(FeatureCalculator):
    """ Implementation of a Calculator for Pandas

    Supported setting keys are the following:

        - `distribution_bins_method`: The method for setting the number of bins when
          calling the `distribution` method. Note that this only has effect when feature is
          either discrete or continuous.
        - `limit_categorical_perc`: The method for truncating categorical variables with
           high cardinality
    """
    def __init__(self):
        super().__init__()
        self.distribution_bins_method = 'sqrt'
        self.limit_categorical_perc = None

    def min(self, column, feature):
        feature.stats['min'] = float(column.min())

    def max(self, column, feature):
        feature.stats['max'] = float(column.max())

    def distribution(self, column, feature, bins=None):
        """ Calculates the histogram for a given feature.

        Args:
            column: Pandas column with the data
            feature: Feature which holds the metadata
            bins: (Only used for numerical features) If a number, the histogram will
                  have `bins` bins. If a string, it will use an automatic NumPy method for
                  estimating this number. See more about available methods here:
                  https://numpy.org/devdocs/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges.
                  If None is provided, it uses the default class' method, which is `sqrt`.
                  For binary features it simply uses bins=2 and for categoricals, bins=|categories| if is not limited
                  with 'limit_categorical_perc' in set_config method.
        """
        if 'no_nan_filtered' not in feature.cache:
            no_na = column.dropna()
            feature.cache['no_nan_filtered'] = no_na
        else:
            no_na = feature.cache['no_nan_filtered']

        if isinstance(feature, (BinaryFeature, CategoricalFeature)):

            no_na = no_na[no_na.isin(feature.stats['domain'])]  # It may be truncated
            t = (no_na.value_counts() / len(no_na)).sort_index()
            feature.stats['distribution'] = t.values
            feature.stats['distribution'] = [float(x) for x in feature.stats['distribution']]
            feature.stats['distribution_bins'] = list(t.index)

        else:
            bins = self.distribution_bins_method if not bins else bins
            histo = np.histogram(no_na, bins=bins)
            feature.stats['distribution'] = list(histo[0] / no_na.count())
            feature.stats['distribution'] = [float(x) for x in feature.stats['distribution']]
            feature.stats['distribution_bins'] = list(histo[1])

    def std(self, column, feature):
        feature.stats['std'] = column.std()

    def mean(self, column, feature):
        feature.stats['mean'] = column.mean()


class SparkStatCalculator(FeatureCalculator):
    def __init__(self):
        pass


class StatCalculatorFactory:
    """ This static class receives a DataFrame and returns a particular implementation
    of a FeatureCalculator
    """
    @classmethod
    def build_calculator(
        cls,
        dataframe: Union["pandas.DataFrame", "pyspark.sql.DataFrame"]  # noqa: F821
    ) -> FeatureCalculator:

        if "pyspark" in str(type(dataframe)):
            raise RuntimeError("Sorry, Pyspark is not supported yet...")

        return PandasStatCalculator()
