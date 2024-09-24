from plotly.offline import plot
import plotly.graph_objects as go
import numpy as np
from pyod.utils.data import generate_data, generate_data_clusters


class DataGenerator:
    def __init__(self):
        """
        >>> data_gen = DataGenerator()
        >>> data_gen.help()

        # Generate Gaussian and Uniform data
        >>> data_gen.generate(n_train=1000, n_test=500, n_features=2, contamination=0.1, random_state=42)
        >>> data_gen.plot_data()

        # Generate cluster data
        >>> data_gen.generate_clusters(n_train=1000, n_test=500, n_clusters=2, n_features=2, contamination=0.1, size='same', density='same', dist=0.25, random_state=42)
        >>> data_gen.plot_data()

        # Generate time series data
        >>> data_gen.generate_time_series(n_samples=1000, n_anomalies=50, trend='linear', seasonality='sine', noise_level=0.1, random_state=42)
        >>> data_gen.plot_data()
        """
        self.X = None

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
        self.X, _, _, _ = generate_data(
            n_train=n_train, n_test=n_test, n_features=n_features,
            contamination=contamination, train_only=train_only,
            offset=offset, behaviour=behaviour, random_state=random_state,
            n_nan=n_nan, n_inf=n_inf
        )

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
        self.X, _, _, _ = generate_data_clusters(
            n_train=n_train, n_test=n_test, n_clusters=n_clusters,
            n_features=n_features, contamination=contamination,
            size=size, density=density, dist=dist, random_state=random_state,
            return_in_clusters=return_in_clusters
        )

    def generate_time_series(self, n_samples=1000, n_anomalies=50, trend='linear',
                             seasonality='sine', noise_level=0.1, random_state=None):
        """
        generate_time_series: Generates synthetic time series data with anomalies, in 2D form (time, value).
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

        # Calcul des données de la série temporelle
        data = trend_data + season_data + noise

        # Injection des anomalies
        anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
        data[anomaly_indices] += np.random.uniform(15, 30, n_anomalies)

        # Création des données en 2D (index, value)
        self.X = np.column_stack((t, data))  # t représente l'index (le temps), data représente les valeurs de la série

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
        if self.X is None:
            print("No data to plot. Please generate data first.")
            return

        # Cas pour les données multidimensionnelles (2D)
        if isinstance(self.X, np.ndarray) and self.X.ndim == 2:
            n_features = self.X.shape[1]

            if n_features == 2:
                # Si on a 2 features, on fait un scatter plot
                trace = go.Scatter(x=self.X[:, 0], y=self.X[:, 1], mode='markers',
                                   marker=dict(size=8, color='blue', line=dict(width=1, color='black')))
                layout = go.Layout(title="2D Data", xaxis=dict(title='Feature 1'), yaxis=dict(title='Feature 2'))
                fig = go.Figure(data=[trace], layout=layout)
                plot(fig, filename='images/raw_data.html')

            else:
                # Si on a plus de 2 dimensions
                print(f"Cannot plot data with {n_features} features. Only 2D data can be visualized.")

        else:
            # Si le format des données est incorrect
            print("Unrecognized data format. Please check your data.")