import numpy as np
from plotly.offline import plot
import plotly.graph_objects as go
from data.data_managing import DataGenerator


class ZScoreAnomalyDetector:
    def __init__(self, threshold=3):
        """
        Z-Score Anomaly Detector.
        Les points avec un Z-score au-delà du seuil sont considérés comme des anomalies.

        :param threshold: seuil pour définir une anomalie (par défaut : 3 écarts-types)
        """
        self.threshold = threshold
        self.anomalies = None

    def predict(self, X):
        """
        Prédit les anomalies basées sur le score Z.

        :param X: numpy array, données de forme (n_samples, n_features) ou (n_samples,)
        :return: numpy array, 1 si anomalie, 0 sinon.
        """
        # Calcul des moyennes et écarts-types par feature
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)

        # Calcul du Z-score
        z_scores = np.abs((X - mean) / std)

        # Si les données sont multidimensionnelles, on prend la somme des Z-scores
        if X.ndim == 2:
            z_scores = np.sum(z_scores, axis=1)

        # Détection des anomalies
        anomalies = (z_scores > self.threshold).astype(int)
        self.anomalies = anomalies

    def plot_results(self, X, file_name_end=""):
        """
        Visualise les résultats avec Plotly si possible.

        :param X: numpy array, données de forme (n_samples, n_features)
        :param file_name_end: str, fin du nom du fichier HTML pour enregistrer le graphique.
        """
        if X.ndim == 2 and X.shape[1] == 2:
            # Données 2D
            trace = go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                               marker=dict(color=self.anomalies, colorscale=[[0, 'green'], [1, 'red']], size=8,
                                           line=dict(width=1, color='black')),
                               name='Data')

            # if all points are anomalies, mention it in the title
            # Same if no anomalies are detected
            if np.all(self.anomalies):
                title = f"DBSCAN Anomaly Detection (2D Data) - All points are anomalies - eps={self.threshold}"
            elif not np.any(self.anomalies):
                title = f"DBSCAN Anomaly Detection (2D Data) - No anomalies detected - eps={self.threshold}"
            else:
                title = f"DBSCAN Anomaly Detection (2D Data) - eps={self.threshold}"

            layout = go.Layout(title=title,
                               xaxis=dict(title='Feature 1'), yaxis=dict(title='Feature 2'))

            fig = go.Figure(data=[trace], layout=layout)
            plot(fig, filename=f'images/z_score_anomalies{"_"+file_name_end if file_name_end else ""}.html')

        else:
            print("Cannot plot data with more than 2 features.")

    @staticmethod
    def help():
        print("""Cette méthode mesure combien d’écarts types un point de données est éloigné de la moyenne. 
        Les points qui s’éloignent beaucoup de la moyenne sont considérés comme des anomalies.""")


class PercentileAnomalyDetector:
    def __init__(self, lower_percentile=5, upper_percentile=95):
        """
        Percentile Anomaly Detector.
        Les points en dessous du seuil du percentile inférieur ou au-dessus du seuil du percentile supérieur
        sont considérés comme des anomalies.

        :param lower_percentile: percentile inférieur pour définir une anomalie (par défaut 5).
        :param upper_percentile: percentile supérieur pour définir une anomalie (par défaut 95).
        """
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.anomalies = None

    def predict(self, X):
        """
        Prédit les anomalies basées sur les percentiles.

        :param X: numpy array, données de forme (n_samples, n_features) ou (n_samples,)
        :return: numpy array, 1 si anomalie, 0 sinon.
        """
        # Calcul des seuils de percentiles
        lower_bound = np.percentile(X, self.lower_percentile, axis=0)
        upper_bound = np.percentile(X, self.upper_percentile, axis=0)

        # Détection des anomalies
        if X.ndim == 2:
            # Pour les données multidimensionnelles, on vérifie pour chaque feature
            mask = np.any((X < lower_bound) | (X > upper_bound), axis=1)
        else:
            # Pour les séries temporelles (1D)
            mask = (X < lower_bound) | (X > upper_bound)

        self.anomalies = mask.astype(int)

    def plot_results(self, X, file_name_end=""):
        """
        Visualise les résultats avec Plotly si possible.

        :param X: numpy array, données de forme (n_samples, n_features)
        :param file_name_end: str, fin du nom du fichier HTML pour enregistrer le graphique.
        """
        if self.anomalies is None:
            print("No anomalies predicted. Please run predict() first.")
            return

        # Données 2D
        if X.ndim == 2 and X.shape[1] == 2:
            trace = go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                               marker=dict(color=self.anomalies, colorscale=[[0, 'green'], [1, 'red']], size=8,
                                           line=dict(width=1, color='black')),
                               name='Data')

            # if all points are anomalies, mention it in the title
            # Same if no anomalies are detected
            if np.all(self.anomalies):
                title = "DBSCAN Anomaly Detection (2D Data) - All points are anomalies - " \
                        f"lower={self.lower_percentile} - upper={self.upper_percentile}"
            elif not np.any(self.anomalies):
                title = "DBSCAN Anomaly Detection (2D Data) - No anomalies detected - " \
                        f"lower={self.lower_percentile} - upper={self.upper_percentile}"
            else:
                title = "DBSCAN Anomaly Detection (2D Data) - " \
                        f"lower={self.lower_percentile} - upper={self.upper_percentile}"

            layout = go.Layout(title=title,
                               xaxis=dict(title='Feature 1'), yaxis=dict(title='Feature 2'))

            fig = go.Figure(data=[trace], layout=layout)
            plot(fig, filename=f'images/percentile_anomalies{"_"+file_name_end if file_name_end else ""}.html')

        else:
            print("Cannot plot data with more than 2 features.")

    @staticmethod
    def help():
        print("""Cette méthode définit des seuils basés sur les percentiles pour identifier les anomalies. 
        Les points en dessous du seuil du percentile inférieur ou au-dessus du seuil du percentile supérieur 
        sont considérés comme des anomalies.""")


if __name__ == "__main__":
    # Générer des données pour tester les méthodes
    data_gen = DataGenerator()

    # Générer des données gaussiennes
    data_gen.generate(n_train=1000, n_test=500, n_features=2, contamination=0.1, random_state=42)
    data_gen.plot_data()

    # Tester la méthode Z-Score
    z_score_detector = ZScoreAnomalyDetector()
    z_score_detector.predict(data_gen.X)
    z_score_detector.plot_results(data_gen.X)

    # Tester la méthode Percentile
    percentile_detector = PercentileAnomalyDetector()
    percentile_detector.predict(data_gen.X)
    percentile_detector.plot_results(data_gen.X)
