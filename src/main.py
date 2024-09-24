from data.data_managing import DataGenerator
from models.StatisticsMethods import ZScoreAnomalyDetector, PercentileAnomalyDetector
from models.MachineLearningMethods import IsolationForestAnomalyDetector, OneClassSVMAnomalyDetector, KNNAnomalyDetector
from models.ProximityBasedMethods import LOFAnomalyDetector, MahalanobisAnomalyDetector
from models.ClusteringMethods import DBSCANAnomalyDetector, KMeansAnomalyDetector

if __name__ == '__main__':

    # Initialisation du générateur de données
    data_gen = DataGenerator()
    # data_gen.help()

    # Liste des jeux de données que nous allons générer
    datasets = {}

    # 1. Génération de données Gaussian et Uniform
    data_gen.generate(n_train=1000, n_test=500, n_features=2, contamination=0.1, random_state=42)
    datasets['gaussian_uniform'] = data_gen.X

    # 2. Génération de données par clusters
    data_gen.generate_clusters(n_train=1000, n_test=500, n_clusters=2, n_features=2, contamination=0.1, size='same',
                               density='same', dist=0.25, random_state=42)
    datasets['clusters'] = data_gen.X

    # 3. Génération de séries temporelles
    data_gen.generate_time_series(n_samples=1000, n_anomalies=50, trend='linear', seasonality='sine', noise_level=0.1,
                                  random_state=42)
    datasets['time_series'] = data_gen.X

    # Appliquer tous les détecteurs d'anomalies sur chaque jeu de données
    detectors = {
        'ZScore': ZScoreAnomalyDetector(),
        'Percentile': PercentileAnomalyDetector(),
        'IsolationForest': IsolationForestAnomalyDetector(),
        'OneClassSVM': OneClassSVMAnomalyDetector(),
        'KNN': KNNAnomalyDetector(),
        'LOF': LOFAnomalyDetector(),
        'Mahalanobis': MahalanobisAnomalyDetector(),
        'DBSCAN': DBSCANAnomalyDetector(),
        'KMeans': KMeansAnomalyDetector()
    }

    results = {}
    for dataset_name, X in datasets.items():
        results[dataset_name] = {}
        for detector_name, detector in detectors.items():
            detector.predict(X)
            detector.plot_results(X, file_name_end=f'{dataset_name}')
