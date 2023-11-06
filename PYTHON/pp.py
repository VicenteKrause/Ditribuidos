import pandas as pd
import numpy as np
import inform_gain as ig
import redundancy_svd as rsvd

# Cargar parámetros desde un archivo CSV
def load_config(file_path='./PYTHON/cnf_sv.csv'):
    params = np.loadtxt(fname=file_path)
    return params

# Cargar datos desde un archivo de datos (por defecto: 'KDDTrain.txt')
def load_data(params, data_file='./PYTHON/KDDTrain.txt'):
    # Definir las categorías de las clases
    class_mapping = {
        'normal': params[3],
        'neptune': params[4],
        'teardrop': params[4],
        'smurf': params[4],
        'pod': params[4],
        'back': params[4],
        'land': params[4],
        'apache2': params[4],
        'processtable': params[4],
        'mailbomb': params[4],
        'udpstorm': params[4],
        'ipsweep': params[5],
        'portsweep': params[5],
        'nmap': params[5],
        'satan': params[5],
        'saint': params[5],
        'mscan': params[5]
    }
    
    # Cargar los datos y realizar preprocesamiento
    data = pd.read_csv(data_file, header=None)
    data = data.drop(42, axis=1)
    data[41].replace(class_mapping, inplace=True)
    data[41] = pd.to_numeric(data[41], errors='coerce')
    
    for i in [1, 2, 3]:
        data[i], _ = pd.factorize(data[i])
    
    # Cargar los índices de muestras desde 'idx_samples.csv'
    idx = np.genfromtxt('./PYTHON/idx_samples.csv', dtype=int) - 1
    data = data.iloc[idx]
    
    # Eliminar filas con valores NaN en la columna 41
    data.dropna(subset=[41], inplace=True)
    
    return data

# Seleccionar características utilizando la ganancia de información y PCA
def select_features(X, params):
    Y = X[41]
    X = X.drop(columns=[41])
    
    feature_indices = []
    gain_values = []
    
    for i in range(len(X.columns)):
        gain_values.append(ig.information_gain(Y, X[i]))
    
    sorted_features = sorted(enumerate(gain_values), key=lambda x: x[1], reverse=True)
    top_k_features = sorted_features[:int(params[1])]
    
    feature_indices, _ = zip(*top_k_features)
    feature_indices = list(feature_indices)
    
    X = X.iloc[:, feature_indices]
    
    V = rsvd.svd_data(X, Y, params)
    
    return gain_values, feature_indices, V

# Guardar los resultados en archivos CSV
def save_results(gain_values, feature_indices, V):
    gain_df = pd.DataFrame(gain_values)
    gain_df.to_csv('gain_values.csv', index=False)
    
    indices_df = pd.DataFrame(feature_indices)
    indices_df.to_csv('feature_indices.csv', index=False)
    
    V_df = pd.DataFrame(V)
    V_df.to_csv('filter_v.csv', index=False)

# Función principal
def main():
    params = load_config()
    data = load_data(params)
    gain_values, feature_indices, V = select_features(data, params)
    save_results(gain_values, feature_indices, V)

if __name__ == '__main__':
    main()