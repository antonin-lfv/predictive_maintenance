from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import numpy as np
from plotly.offline import plot
import plotly.graph_objects as go
from data.data_managing import DataGenerator


class DBSCANAnomalyDetector:
    def __init__(self, eps=0.5, min_samples=5):
        """
        DBSCAN Anomaly Detector.
        Regroupe les points par densité et détecte les points isolés comme anomalies.

        :param eps: Distance maximale entre deux points pour qu'ils soient considérés dans le même cluster.
        :param min_samples: Nombre minimum de points requis pour former un cluster.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        self.anomalies = None

    def predict(self, X):
        """
        Prédit les anomalies en utilisant DBSCAN.

        :param X: numpy array, données de forme (n_samples, n_features) ou (n_samples,)
        :return: numpy array, 1 si anomalie, 0 sinon.
        """
        # Apprentissage du modèle et clustering
        labels = self.model.fit_predict(X)

        # Les points ayant le label -1 sont considérés comme des anomalies
        self.anomalies = (labels == -1).astype(int)

    def plot_results(self, X, file_name_end=""):
        """
        Visualise les résultats avec Plotly si possible (en 1D ou 2D).

        :param X: numpy array, données de forme (n_samples, n_features)
        :param file_name_end: str, fin du nom du fichier HTML pour enregistrer le graphique.
        """
        if self.anomalies is None:
            print("No anomalies predicted. Please run predict() first.")
            return

        # Cas des données 2D
        if X.ndim == 2 and X.shape[1] == 2:
            trace = go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                               marker=dict(color=self.anomalies, colorscale=[[0, 'green'], [1, 'red']], size=8,
                                           line=dict(width=1, color='black')),
                               name='Data')

            # if all points are anomalies, mention it in the title
            # Same if no anomalies are detected
            if np.all(self.anomalies):
                title = f"DBSCAN Anomaly Detection (2D Data) - All points are anomalies - eps={self.eps}, " \
                        f"min_samples={self.min_samples}"
            elif not np.any(self.anomalies):
                title = f"DBSCAN Anomaly Detection (2D Data) - No anomalies detected - eps={self.eps}, " \
                        f"min_samples={self.min_samples}"
            else:
                title = f"DBSCAN Anomaly Detection (2D Data) - eps={self.eps}, " \
                        f"min_samples={self.min_samples}"

            layout = go.Layout(title=title,
                               xaxis=dict(title='Feature 1'), yaxis=dict(title='Feature 2'))

            fig = go.Figure(data=[trace], layout=layout)
            plot(fig, filename=f'images/dbscan_anomalies{"_"+file_name_end if file_name_end else ""}.html')

        else:
            print("Cannot plot data with more than 2 features.")

    @staticmethod
    def help():
        print("""DBSCAN (Density-Based Spatial Clustering of Applications with Noise) : Regroupe les points de données 
        en fonction de leur densité, les points qui n’appartiennent à aucun cluster étant considérés comme 
        des valeurs aberrantes.""")


class KMeansAnomalyDetector:
    def __init__(self, n_clusters=3, threshold=None):
        """
        K-Means Clustering Anomaly Detector.
        Identifie les anomalies en fonction de la distance des points aux centres de clusters.

        :param n_clusters: Nombre de clusters à former (par défaut 3).
        :param threshold: Seuil de distance au-dessus duquel un point est considéré comme une anomalie.
                          Si None, utilise un seuil basé sur les percentiles.
        """
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, n_init='auto')
        self.threshold = threshold
        self.anomalies = None
        self.distances = None

    def predict(self, X):
        """
        Prédit les anomalies en utilisant K-Means.

        :param X: numpy array, données de forme (n_samples, n_features)
        :return: numpy array, 1 si anomalie, 0 sinon.
        """
        # Apprentissage du modèle K-Means et prédiction des clusters
        self.model.fit(X)
        labels = self.model.predict(X)

        # Calcul des distances des points aux centres des clusters
        centers = self.model.cluster_centers_
        self.distances = np.linalg.norm(X - centers[labels], axis=1)

        # Si aucun seuil n'est défini, on prend le 95e percentile des distances comme seuil
        if self.threshold is None:
            self.threshold = np.percentile(self.distances, 95)

        # Les points dont la distance est supérieure au seuil sont des anomalies
        self.anomalies = (self.distances > self.threshold).astype(int)

    def plot_results(self, X, file_name_end=""):
        """
        Visualise les résultats avec Plotly si possible (en 1D ou 2D).

        :param X: numpy array, données de forme (n_samples, n_features)
        :param file_name_end: str, fin du nom du fichier HTML pour enregistrer le graphique.
        """
        if self.anomalies is None:
            print("No anomalies predicted. Please run predict() first.")
            return

        # Cas des données 2D
        if X.ndim == 2 and X.shape[1] == 2:
            trace = go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                               marker=dict(color=self.anomalies, colorscale=[[0, 'green'], [1, 'red']], size=8,
                                           line=dict(width=1, color='black')),
                               name='Data')

            # if all points are anomalies, mention it in the title
            # Same if no anomalies are detected
            if np.all(self.anomalies):
                title = f"DBSCAN Anomaly Detection (2D Data) - All points are anomalies - eps={self.threshold}" \
                        f" - n_clusters={self.n_clusters}"
            elif not np.any(self.anomalies):
                title = f"DBSCAN Anomaly Detection (2D Data) - No anomalies detected - eps={self.threshold}" \
                        f" - n_clusters={self.n_clusters}"
            else:
                title = f"DBSCAN Anomaly Detection (2D Data) - eps={self.threshold}" \
                        f" - n_clusters={self.n_clusters}"

            layout = go.Layout(title=title,
                               xaxis=dict(title='Feature 1'), yaxis=dict(title='Feature 2'))

            fig = go.Figure(data=[trace], layout=layout)
            plot(fig, filename=f'images/kmeans_anomalies{"_"+file_name_end if file_name_end else ""}.html')

        else:
            print("Cannot plot data with more than 2 features.")

    @staticmethod
    def help():
        print("""K-Means : Algorithme de clustering qui partitionne les données en K clusters en fonction de la distance 
        des points aux centres de clusters.""")


if __name__ == "__main__":
    # Générer des données pour tester les méthodes
    data_gen = DataGenerator()

    # Générer des données gaussiennes
    data_gen.generate_time_series(n_samples=1000, n_anomalies=50, trend='linear', seasonality='sine', noise_level=0.1,
                                  random_state=42)
    # data_gen.plot_data()

    # DBSCAN
    dbscan = DBSCANAnomalyDetector(eps=0.5, min_samples=5)
    dbscan.predict(data_gen.X)
    dbscan.plot_results(data_gen.X)

    # K-Means
    kmeans = KMeansAnomalyDetector(n_clusters=3)
    kmeans.predict(data_gen.X)
    kmeans.plot_results(data_gen.X)
