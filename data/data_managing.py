import matplotlib.pyplot as plt
import numpy as np
from pyod.utils.data import generate_data, generate_data_categorical, generate_data_clusters


class DataGenerator:
    def __init__(self):
        """
        >>> data_gen = DataGenerator()
        >>> data_gen.help()

        # Generate Gaussian and Uniform data
        >>> data_gen.generate(n_train=1000, n_test=500, n_features=2, contamination=0.1, random_state=42)
        >>> data_gen.plot_data()

        # Generate categorical data
        >>> data_gen.generate_categorical(n_train=1000, n_test=500, n_features=2, n_informative=2, n_category_in=2, n_category_out=2, contamination=0.1, random_state=42)
        >>> data_gen.plot_data()

        # Generate cluster data
        >>> data_gen.generate_clusters(n_train=1000, n_test=500, n_clusters=2, n_features=2, contamination=0.1, size='same', density='same', dist=0.25, random_state=42)
        >>> data_gen.plot_data()

        # Generate time series data
        >>> data_gen.generate_time_series(n_samples=1000, n_anomalies=50, trend='linear', seasonality='sine', noise_level=0.1, random_state=42)
        >>> data_gen.plot_data()
        """
        self.time_series_labels = None
        self.time_series = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def generate(self, n_train=1000, n_test=500, n_features=2, contamination=0.1,
                 train_only=False, offset=10, behaviour='new', random_state=None,
                 n_nan=0, n_inf=0):
        """
        generate: Generates synthesized data with Gaussian and Uniform distributions.
            Parameters:
            - n_train: int (default=1000), Number of training points to generate.
            - n_test: int (default=500), Number of test points to generate.
            - n_features: int (default=2), Number of features.
            - contamination: float (default=0.1), Proportion of outliers.
            - train_only: bool (default=False), Generate train data only.
            - offset: int (default=10), Adjust value range.
            - behaviour: str (default='new'), 'old' or 'new' behaviour.
            - random_state: int or None (default=None), Seed for random generator.
            - n_nan: int (default=0), Number of NaN values.
            - n_inf: int (default=0), Number of Inf values.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=n_train, n_test=n_test, n_features=n_features,
            contamination=contamination, train_only=train_only,
            offset=offset, behaviour=behaviour, random_state=random_state,
            n_nan=n_nan, n_inf=n_inf
        )
        # reset the time series data
        self.time_series = None
        self.time_series_labels = None

    def generate_categorical(self, n_train=1000, n_test=500, n_features=2, n_informative=2,
                             n_category_in=2, n_category_out=2, contamination=0.1,
                             shuffle=True, random_state=None):
        """
        generate_categorical: Generates synthesized categorical data.
            Parameters:
            - n_train: int (default=1000), Number of training points to generate.
            - n_test: int (default=500), Number of test points to generate.
            - n_features: int (default=2), Number of features.
            - n_informative: int (default=2), Number of informative features.
            - n_category_in: int (default=2), Number of categories in inliers.
            - n_category_out: int (default=2), Number of categories in outliers.
            - contamination: float (default=0.1), Proportion of outliers.
            - shuffle: bool (default=True), Shuffle inliers.
            - random_state: int or None (default=None), Seed for random generator.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data_categorical(
            n_train=n_train, n_test=n_test, n_features=n_features,
            n_informative=n_informative, n_category_in=n_category_in,
            n_category_out=n_category_out, contamination=contamination,
            shuffle=shuffle, random_state=random_state
        )
        # reset the time series data
        self.time_series = None
        self.time_series_labels = None

    def generate_clusters(self, n_train=1000, n_test=500, n_clusters=2, n_features=2,
                          contamination=0.1, size='same', density='same', dist=0.25,
                          random_state=None, return_in_clusters=False):
        """
        generate_clusters: Generates synthesized data in clusters.
            Parameters:
            - n_train: int (default=1000), Number of training points to generate.
            - n_test: int (default=500), Number of test points to generate.
            - n_clusters: int (default=2), Number of clusters.
            - n_features: int (default=2), Number of features.
            - contamination: float (default=0.1), Proportion of outliers.
            - size: str (default='same'), Size of each cluster ('same' or 'different').
            - density: str (default='same'), Density of each cluster ('same' or 'different').
            - dist: float (default=0.25), Distance between clusters.
            - random_state: int or None (default=None), Seed for random generator.
            - return_in_clusters: bool (default=False), Return clusters separately.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data_clusters(
            n_train=n_train, n_test=n_test, n_clusters=n_clusters,
            n_features=n_features, contamination=contamination,
            size=size, density=density, dist=dist, random_state=random_state,
            return_in_clusters=return_in_clusters
        )
        # reset the time series data
        self.time_series = None
        self.time_series_labels = None

    def generate_time_series(self, n_samples=1000, n_anomalies=50, trend='linear',
                             seasonality='sine', noise_level=0.1, random_state=None):
        """
        generate_time_series: Generates synthetic time series data with anomalies.
            Parameters:
            - n_samples: int (default=1000), Number of samples in the time series.
            - n_anomalies: int (default=50), Number of anomalies to inject.
            - trend: str (default='linear'), Type of trend ('linear', 'quadratic', or 'none').
            - seasonality: str (default='sine'), Type of seasonality ('sine', 'cosine', or 'none').
            - noise_level: float (default=0.1), Level of noise.
            - random_state: int or None (default=None), Seed for random generator.
        """
        np.random.seed(random_state)
        t = np.arange(n_samples)

        if trend == 'linear':
            trend_data = t * 0.05
        elif trend == 'quadratic':
            trend_data = 0.0005 * (t ** 2)
        else:
            trend_data = np.zeros(n_samples)

        if seasonality == 'sine':
            season_data = np.sin(t * 0.1) * 10
        elif seasonality == 'cosine':
            season_data = np.cos(t * 0.1) * 10
        else:
            season_data = np.zeros(n_samples)

        noise = np.random.normal(0, noise_level, n_samples)

        self.time_series = trend_data + season_data + noise
        self.time_series_labels = np.zeros(n_samples)

        anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
        self.time_series[anomaly_indices] += np.random.uniform(15, 30, n_anomalies)
        self.time_series_labels[anomaly_indices] = 1

    @staticmethod
    def help():
        docstring = """
        DataGenerator Class - Generate synthetic data for anomaly detection.

        Methods:
        generate: Generates synthesized data with Gaussian and Uniform distributions.
            Parameters:
            - n_train: int (default=1000), Number of training points to generate.
            - n_test: int (default=500), Number of test points to generate.
            - n_features: int (default=2), Number of features.
            - contamination: float (default=0.1), Proportion of outliers.
            - train_only: bool (default=False), Generate train data only.
            - offset: int (default=10), Adjust value range.
            - behaviour: str (default='new'), 'old' or 'new' behaviour.
            - random_state: int or None (default=None), Seed for random generator.
            - n_nan: int (default=0), Number of NaN values.
            - n_inf: int (default=0), Number of Inf values.

        generate_categorical: Generates synthesized categorical data.
            Parameters:
            - n_train: int (default=1000), Number of training points to generate.
            - n_test: int (default=500), Number of test points to generate.
            - n_features: int (default=2), Number of features.
            - n_informative: int (default=2), Number of informative features.
            - n_category_in: int (default=2), Number of categories in inliers.
            - n_category_out: int (default=2), Number of categories in outliers.
            - contamination: float (default=0.1), Proportion of outliers.
            - shuffle: bool (default=True), Shuffle inliers.
            - random_state: int or None (default=None), Seed for random generator.

        generate_clusters: Generates synthesized data in clusters.
            Parameters:
            - n_train: int (default=1000), Number of training points to generate.
            - n_test: int (default=500), Number of test points to generate.
            - n_clusters: int (default=2), Number of clusters.
            - n_features: int (default=2), Number of features.
            - contamination: float (default=0.1), Proportion of outliers.
            - size: str (default='same'), Size of each cluster ('same' or 'different').
            - density: str (default='same'), Density of each cluster ('same' or 'different').
            - dist: float (default=0.25), Distance between clusters.
            - random_state: int or None (default=None), Seed for random generator.
            - return_in_clusters: bool (default=False), Return clusters separately.
        
        generate_time_series: Generates synthetic time series data with anomalies.
            Parameters:
            - n_samples: int (default=1000), Number of samples in the time series.
            - n_anomalies: int (default=50), Number of anomalies to inject.
            - trend: str (default='linear'), Type of trend ('linear', 'quadratic', or 'none').
            - seasonality: str (default='sine'), Type of seasonality ('sine', 'cosine', or 'none').
            - noise_level: float (default=0.1), Level of noise.
            - random_state: int or None (default=None), Seed for random generator.

        plot_data: Plots the generated data if it exists.
        """
        print(docstring)

    def plot_data(self):
        if self.time_series is not None:
            plt.figure(figsize=(14, 6))
            plt.plot(self.time_series, label='Time Series Data')
            plt.scatter(np.where(self.time_series_labels == 1)[0], self.time_series[self.time_series_labels == 1],
                        color='red', label='Anomalies', marker='x')
            plt.title("Time Series Data with Anomalies")
            plt.legend()
            plt.show()
        else:
            if self.X_train is None or self.X_test is None:
                print("No data to plot. Please generate data first.")
                return

            plt.figure(figsize=(14, 6))

            plt.subplot(1, 2, 1)
            plt.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train, cmap='viridis', marker='o',
                        edgecolor='k')
            plt.title("Training Data")

            plt.subplot(1, 2, 2)
            plt.scatter(self.X_test[:, 0], self.X_test[:, 1], c=self.y_test, cmap='viridis', marker='o', edgecolor='k')
            plt.title("Test Data")

            plt.show()
