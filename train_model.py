import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Membaca dataset
df = pd.read_csv('dataset/angka_calonpembelimobil.csv')

# Memisahkan fitur (X) dan target (y)
X = df[['Usia', 'Status', 'Kelamin', 'Penghasilan', 'Memiliki_Mobil']]
y = df['Beli_Mobil']  # Target perkiraan

# Membagi data menjadi data latih (training set) dan data uji (test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi model regresi linear
linear_regressor = LinearRegression()

# Melatih model menggunakan data latih
linear_regressor.fit(X_train, y_train)

# Menyimpan model ke dalam file menggunakan pickle
with open('regression_linear_model.pkl', 'wb') as model_file:
    pickle.dump(linear_regressor, model_file)

print("Model telah berhasil dilatih dan disimpan ke 'regression_linear_model.pkl'.")