import numpy as np
from plotly.offline import plot
import plotly.graph_objects as go
from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv
from sklearn.neighbors import LocalOutlierFactor
from data.data_managing import DataGenerator


class MahalanobisAnomalyDetector:
    def __init__(self, threshold=None):
        """
        Mahalanobis Distance Anomaly Detector.
        Identifie les anomalies en fonction de la distance de Mahalanobis par rapport à la distribution des données.

        :param threshold: Seuil de distance au-dessus duquel un point est considéré comme une anomalie.
                          Si None, utilise un seuil basé sur les percentiles.
        """
        self.threshold = threshold
        self.anomalies = None
        self.distances = None
        self.mean = None
        self.cov_inv = None

    def predict(self, X):
        """
        Prédit les anomalies en fonction de la distance de Mahalanobis.

        :param X: numpy array, données de forme (n_samples, n_features)
        :return: numpy array, 1 si anomalie, 0 sinon.
        """
        # Calcul de la moyenne et de l'inverse de la matrice de covariance
        self.mean = np.mean(X, axis=0)
        cov_matrix = np.cov(X, rowvar=False)
        self.cov_inv = inv(cov_matrix)

        # Calcul des distances de Mahalanobis pour chaque point
        self.distances = np.array([mahalanobis(x, self.mean, self.cov_inv) for x in X])

        # Si aucun seuil n'est fourni, on prend le 95e percentile des distances comme seuil
        if self.threshold is None:
            self.threshold = np.percentile(self.distances, 95)

        # Détection des anomalies
        self.anomalies = (self.distances > self.threshold).astype(int)

    def plot_results(self, X, file_name_end=""):
        """
        Visualise les résultats avec Plotly si possible.

        :param X: numpy array, données de forme (n_samples, n_features)
        :param file_name_end: str, fin du nom du fichier HTML pour enregistrer le graphique.
        """
        if self.anomalies is None:
            print("No anomalies predicted. Please run predict() first.")
            return

        # Cas pour les données 2D
        if X.ndim == 2 and X.shape[1] == 2:
            trace = go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                               marker=dict(color=self.anomalies, colorscale=[[0, 'green'], [1, 'red']], size=8,
                                           line=dict(width=1, color='black')),
                               name='Data')

            # if all points are anomalies, mention it in the title
            # Same if no anomalies are detected
            if np.all(self.anomalies):
                title = f"DBSCAN Anomaly Detection (2D Data) - All points are anomalies - eps={self.threshold}"
            elif not np.any(self.anomalies):
                title = "DBSCAN Anomaly Detection (2D Data) - No anomalies detected - eps={self.threshold}"
            else:
                title = "DBSCAN Anomaly Detection (2D Data) - eps={self.threshold}"

            layout = go.Layout(title=title,
                               xaxis=dict(title='Feature 1'), yaxis=dict(title='Feature 2'))

            fig = go.Figure(data=[trace], layout=layout)
            plot(fig, filename=f'images/mahalanobis_anomalies{"_"+file_name_end if file_name_end else ""}.html')

        else:
            print("Cannot plot data with more than 2 features.")

    @staticmethod
    def help():
        print("""Mesure la distance des points de données par rapport au centre de la distribution des données, 
        en tenant compte des corrélations entre les caractéristiques""")


class LOFAnomalyDetector:
    def __init__(self, n_neighbors=20):
        """
        Local Outlier Factor Anomaly Detector.
        Identifie les anomalies en fonction de la densité locale comparée à celle des voisins.

        :param n_neighbors: Nombre de voisins utilisés pour calculer le facteur LOF (par défaut 20).
        """
        self.n_neighbors = n_neighbors
        self.model = LocalOutlierFactor(n_neighbors=n_neighbors)
        self.anomalies = None
        self.scores = None

    def predict(self, X):
        """
        Prédit les anomalies en utilisant le Local Outlier Factor (LOF).

        :param X: numpy array, données de forme (n_samples, n_features)
        :return: numpy array, 1 si anomalie, 0 sinon.
        """
        # Calcul du facteur LOF pour chaque point
        self.scores = self.model.fit_predict(X)

        # LOF renvoie -1 pour les anomalies et 1 pour les points normaux
        self.anomalies = (self.scores == -1).astype(int)

    def plot_results(self, X, file_name_end=""):
        """
        Visualise les résultats avec Plotly si possible.

        :param X: numpy array, données de forme (n_samples, n_features)
        :param file_name_end: str, fin du nom du fichier HTML pour enregistrer le graphique.
        """
        if self.anomalies is None:
            print("No anomalies predicted. Please run predict() first.")
            return

        # Cas pour les données 2D
        if X.ndim == 2 and X.shape[1] == 2:
            trace = go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                               marker=dict(color=self.anomalies, colorscale=[[0, 'green'], [1, 'red']], size=8,
                                           line=dict(width=1, color='black')),
                               name='Data')

            # if all points are anomalies, mention it in the title
            # Same if no anomalies are detected
            if np.all(self.anomalies):
                title = (f"DBSCAN Anomaly Detection (2D Data) - All points are anomalies - "
                         f"n_neighbors={self.n_neighbors}")
            elif not np.any(self.anomalies):
                title = f"DBSCAN Anomaly Detection (2D Data) - No anomalies detected - n_neighbors={self.n_neighbors}"
            else:
                title = f"DBSCAN Anomaly Detection (2D Data) - n_neighbors={self.n_neighbors}"

            layout = go.Layout(title=title,
                               xaxis=dict(title='Feature 1'), yaxis=dict(title='Feature 2'))

            fig = go.Figure(data=[trace], layout=layout)
            plot(fig, filename=f'images/lof_anomalies{"_"+file_name_end if file_name_end else ""}.html')

        else:
            print("Cannot plot data with more than 2 features.")

    @staticmethod
    def help():
        print("""Identifie les anomalies en fonction de la densité locale comparée à celle des voisins.""")


if __name__ == "__main__":
    # Générer des données pour tester les méthodes
    data_gen = DataGenerator()

    # Générer des données gaussiennes
    data_gen.generate_clusters(n_train=1000, n_test=500, n_features=2, contamination=0.1, random_state=42)
    data_gen.plot_data()

    # Tester la détection d'anomalies avec la distance de Mahalanobis
    mahalanobis_detector = MahalanobisAnomalyDetector()
    mahalanobis_detector.predict(data_gen.X)
    mahalanobis_detector.plot_results(data_gen.X)

    # Tester la détection d'anomalies avec le facteur LOF
    lof_detector = LOFAnomalyDetector()
    lof_detector.predict(data_gen.X)
    lof_detector.plot_results(data_gen.X)
