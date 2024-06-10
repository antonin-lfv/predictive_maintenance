from data.data_managing import DataGenerator

data_gen = DataGenerator()
data_gen.help()

# Generate Gaussian and Uniform data
data_gen.generate(n_train=1000, n_test=500, n_features=2, contamination=0.1, random_state=42)
data_gen.plot_data()

# Generate categorical data
data_gen.generate_categorical(n_train=1000, n_test=500, n_features=2, n_informative=2, n_category_in=2,
                              n_category_out=2, contamination=0.1, random_state=42)
data_gen.plot_data()

# Generate cluster data
data_gen.generate_clusters(n_train=1000, n_test=500, n_clusters=2, n_features=2, contamination=0.1, size='same',
                           density='same', dist=0.25, random_state=42)
data_gen.plot_data()
