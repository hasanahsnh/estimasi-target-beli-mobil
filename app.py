from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error, confusion_matrix
from sklearn.model_selection import train_test_split

app = Flask(__name__, template_folder='template')

# Memuat model yang telah dilatih sebelumnya
with open('regression_linear_model.pkl', 'rb') as model_file:
    linear_regressor = pickle.load(model_file)

# Memuat data uji untuk menghitung error
df = pd.read_csv('dataset/data-fix-calon-pembeli-mobil.csv')
X = df[['Usia', 'Status', 'Kelamin', 'Memiliki_Mobil', 'Penghasilan']]
y = df['Beli_Mobil']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    global X_test, y_test

    # Menerima input dari form
    usia = int(request.form['Usia'])
    status = int(request.form['Status'])
    kelamin = int(request.form['Kelamin'])
    memiliki_mobil = int(request.form['Memiliki_Mobil'])
    penghasilan = int(request.form['Penghasilan'])
    
    # Membuat dataframe dengan input
    new_data = pd.DataFrame({
        'Usia': [usia],
        'Status': [status],
        'Kelamin': [kelamin],
        'Memiliki_Mobil': [memiliki_mobil],
        'Penghasilan': [penghasilan]
    })
    
    # Melakukan prediksi jumlah mobil yang dimiliki
    predicted_buy_of_car = linear_regressor.predict(new_data)[0]
    
    # Membulatkan hasil prediksi
    rounded_prediction = round(predicted_buy_of_car)
    
    # Menentukan hasil prediksi
    if rounded_prediction == 1:
        hasil_prediksi = "Beli Mobil"
    else:
        hasil_prediksi = "Tidak Beli Mobil"
        
    # Menghitung error menggunakan data uji
    y_pred = linear_regressor.predict(X_test)
    y_pred_rounded = [round(pred) for pred in y_pred]
    
    # Menghitung confusion matrix
    cm = confusion_matrix(y_test, y_pred_rounded)
    
    # Menghitung mean absolute error (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Menghitung miss classification error
    miss_classification_error = (cm[0][1] + cm[1][0]) / sum(sum(cm))
    
    # Menghitung relative foreground error (sederhana)
    foreground_error = (cm[0][0] + cm[1][1]) / sum(sum(cm))
    
    # Menyimpan hasil prediksi dan error dalam dictionary
    prediction_results = {
        'predicted_value': rounded_prediction,
        'hasil_prediksi': hasil_prediksi,
        'mean_absolute_error': mae,
        'miss_classification_error': miss_classification_error,
        'foreground_error': foreground_error
    }

    # Merender template HTML dengan hasil prediksi
    return render_template('index.html', prediction_results=prediction_results)

if __name__ == '__main__':
    app.run(debug=True)
