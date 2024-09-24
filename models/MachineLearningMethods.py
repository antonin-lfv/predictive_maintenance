from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import NearestNeighbors
import numpy as np
from plotly.offline import plot
import plotly.graph_objects as go
from data.data_managing import DataGenerator


class IsolationForestAnomalyDetector:
    def __init__(self, n_estimators=100, contamination=0.1, random_state=None):
        """
        Isolation Forest Anomaly Detector.
        Identifie les anomalies en utilisant l'Isolation Forest.

        :param n_estimators: Nombre d'arbres dans la forêt (par défaut 100).
        :param contamination: Proportion d'anomalies attendue (par défaut 0.1).
        :param random_state: Graine pour assurer la reproductibilité des résultats.
        """
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=random_state)
        self.anomalies = None

    def predict(self, X):
        """
        Prédit les anomalies en utilisant Isolation Forest.

        :param X: numpy array, données de forme (n_samples, n_features) ou (n_samples,)
        :return: numpy array, 1 si anomalie, 0 sinon.
        """
        # Apprentissage du modèle et prédiction
        self.model.fit(X)
        predictions = self.model.predict(X)

        # L'Isolation Forest renvoie -1 pour les anomalies et 1 pour les points normaux, on convertit cela en 0 et 1
        self.anomalies = (predictions == -1).astype(int)

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
                title = f"DBSCAN Anomaly Detection (2D Data) - All points are anomalies - eps={self.n_estimators}, " \
                        f"contamination={self.contamination}"
            elif not np.any(self.anomalies):
                title = "DBSCAN Anomaly Detection (2D Data) - No anomalies detected - eps={self.n_estimators}, " \
                        f"contamination={self.contamination}"
            else:
                title = "DBSCAN Anomaly Detection (2D Data) - eps={self.n_estimators}, " \
                        f"contamination={self.contamination}"

            layout = go.Layout(title=title,
                               xaxis=dict(title='Feature 1'), yaxis=dict(title='Feature 2'))

            fig = go.Figure(data=[trace], layout=layout)
            plot(fig, filename=f'images/isolation_forest_anomalies{"_"+file_name_end if file_name_end else ""}.html')

        else:
            print("Cannot plot data with more than 2 features.")

    @staticmethod
    def help():
        print("""Une méthode d’apprentissage ensembliste qui construit une structure arborescente pour isoler 
        efficacement les anomalies.""")


class OneClassSVMAnomalyDetector:
    def __init__(self, kernel='rbf', nu=0.1, gamma='scale'):
        """
        One-Class SVM Anomaly Detector.
        Identifie les anomalies en utilisant un SVM One-Class.

        :param kernel: Type de noyau à utiliser (par défaut 'rbf').
        :param nu: Paramètre de sensibilité pour définir la proportion d'anomalies (par défaut 0.1).
        :param gamma: Paramètre du noyau (par défaut 'scale').
        """
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
        self.anomalies = None

    def predict(self, X):
        """
        Prédit les anomalies en utilisant One-Class SVM.

        :param X: numpy array, données de forme (n_samples, n_features) ou (n_samples,)
        :return: numpy array, 1 si anomalie, 0 sinon.
        """
        # Apprentissage du modèle et prédiction
        self.model.fit(X)
        predictions = self.model.predict(X)

        # One-Class SVM renvoie -1 pour les anomalies et 1 pour les points normaux, on convertit cela en 0 et 1
        self.anomalies = (predictions == -1).astype(int)

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
                title = f"DBSCAN Anomaly Detection (2D Data) - All points are anomalies - kernel={self.kernel}, " \
                        f"nu={self.nu}, gamma={self.gamma}"
            elif not np.any(self.anomalies):
                title = "DBSCAN Anomaly Detection (2D Data) - No anomalies detected - kernel={self.kernel}, " \
                        f"nu={self.nu}, gamma={self.gamma}"
            else:
                title = "DBSCAN Anomaly Detection (2D Data) - kernel={self.kernel}, " \
                        f"nu={self.nu}, gamma={self.gamma}"

            layout = go.Layout(title=title,
                               xaxis=dict(title='Feature 1'), yaxis=dict(title='Feature 2'))

            fig = go.Figure(data=[trace], layout=layout)
            plot(fig, filename=f'images/one_class_svm_anomalies{"_"+file_name_end if file_name_end else ""}.html')

        else:
            print("Cannot plot data with more than 2 features.")

    @staticmethod
    def help():
        print("""Un algorithme d'apprentissage non supervisé qui apprend une frontière de décision pour les données 
        normales et identifie les anomalies comme des points qui sont loin de cette frontière.""")


class KNNAnomalyDetector:
    def __init__(self, n_neighbors=5):
        """
        K-Nearest Neighbors Anomaly Detector.
        Détecte les anomalies en fonction de la distance aux K plus proches voisins.

        :param n_neighbors: Nombre de voisins à considérer (par défaut 5).
        """
        self.n_neighbors = n_neighbors
        self.model = NearestNeighbors(n_neighbors=n_neighbors)
        self.distances = None
        self.anomalies = None

    def predict(self, X, threshold=None):
        """
        Prédit les anomalies en fonction des distances aux K plus proches voisins.

        :param X: numpy array, données de forme (n_samples, n_features) ou (n_samples,)
        :param threshold: Seuil de distance au-dessus duquel un point est considéré comme anomalie.
                          Si None, utilise un seuil basé sur les percentiles.
        :return: numpy array, 1 si anomalie, 0 sinon.
        """
        # Entraînement du modèle Nearest Neighbors
        self.model.fit(X)

        # Calcul des distances des K plus proches voisins
        distances, _ = self.model.kneighbors(X)

        # On s'intéresse à la distance du Kème plus proche voisin pour chaque point
        self.distances = distances[:, -1]

        # Si aucun seuil n'est fourni, on prend le 95e percentile des distances comme seuil
        if threshold is None:
            threshold = np.percentile(self.distances, 95)

        # Détection des anomalies
        self.anomalies = (self.distances > threshold).astype(int)

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
                title = f"DBSCAN Anomaly Detection (2D Data) - All points are anomalies - K={self.n_neighbors}"
            elif not np.any(self.anomalies):
                title = f"DBSCAN Anomaly Detection (2D Data) - No anomalies detected - K={self.n_neighbors}"
            else:
                title = f"DBSCAN Anomaly Detection (2D Data) - K={self.n_neighbors}"

            layout = go.Layout(title=title,
                               xaxis=dict(title='Feature 1'), yaxis=dict(title='Feature 2'))

            fig = go.Figure(data=[trace], layout=layout)
            plot(fig, filename=f'images/knn_anomalies{"_"+file_name_end if file_name_end else ""}.html')

        else:
            print("Cannot plot data with more than 2 features.")

    @staticmethod
    def help():
        print("""Un algorithme simple qui identifie les anomalies en fonction de la distance 
        aux K plus proches voisins.""")


if __name__ == "__main__":
    # Générer des données pour tester les méthodes
    data_gen = DataGenerator()

    # Générer des données gaussiennes
    data_gen.generate(n_train=1000, n_test=500, n_features=2, contamination=0.1, random_state=42)
    data_gen.plot_data()

    # Tester les méthodes sur les données générées
    iforest = IsolationForestAnomalyDetector()
    iforest.predict(data_gen.X)
    iforest.plot_results(data_gen.X)

    ocsvm = OneClassSVMAnomalyDetector()
    ocsvm.predict(data_gen.X)
    ocsvm.plot_results(data_gen.X)

    knn = KNNAnomalyDetector(n_neighbors=5)
    knn.predict(data_gen.X)
    knn.plot_results(data_gen.X)
