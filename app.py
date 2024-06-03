from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__, template_folder='template')

# Memuat model yang telah dilatih sebelumnya
with open('regression_linear_model.pkl', 'rb') as model_file:
    linear_regressor = pickle.load(model_file)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
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
        
        
    # Menyimpan hasil prediksi dalam dictionary
    prediction_results = {
        'predicted_value': rounded_prediction,
        'hasil_prediksi': hasil_prediksi
    }

    # Merender template HTML dengan hasil prediksi
    return render_template('index.html', prediction_results=prediction_results)

if __name__ == '__main__':
    app.run(debug=True)
