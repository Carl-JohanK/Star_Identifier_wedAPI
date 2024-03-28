from flask import Flask, request, json
import pickle
import numpy as np

app = Flask(__name__)

ml_model = pickle.load(open('ML_code/Identify_Stars/star_dt_clr.sav', 'rb'))
model_accuracy = pickle.load(open('ML_code/Identify_Stars/star_accuracy.sav', 'rb'))

history = []

@app.route("/star/predict")
def predict():
    temperature = float(request.json["Temp"]) 
    luminosity = float(request.json["Lumin"])
    radius = float(request.json["Radius"])
    color = request.json["Color"]
    spectral_c = request.json["SC"] #Spectral_Class

    color = color.lower()
    spectral_c = spectral_c.upper()
    
    if color == "red":
        color = 5
    elif color == "blue":
        color = 0
    elif color == "blue white":
        color = 1
    elif color == "orange":
        color = 2
    elif color == "orange red":
        color = 3
    elif color == "pale yellow orange":
        color = 4
    elif color == "white":
        color = 6
    elif color == "yellowish white":
        color = 7
    elif color == "yellowish":
        color = 8
    else:
        return "invalid color"
    
    if spectral_c == "A":
        spectral_c = 0
    elif spectral_c == "B":
        spectral_c = 1
    elif spectral_c == "F":
        spectral_c = 2
    elif spectral_c == "G":
        spectral_c = 3
    elif spectral_c == "K":
        spectral_c = 4
    elif spectral_c == "M":
        spectral_c = 5
    elif spectral_c == "O":
        spectral_c = 6
    else:
        return "invalid class"
    
    color = float(color)
    spectral_c = float(spectral_c)

    predict_vals = [temperature, luminosity, radius, color, spectral_c]

    prediction = ml_model.predict([predict_vals])
    accuracy = model_accuracy

    history.append([predict_vals, prediction, accuracy])

    return "the predicted value:" + str(prediction)  + " the Accuracy score:" + str(accuracy) + "\nRed Dwarf - 0 | Brown Dwarf - 1 \n White Dwarf - 2 | Main Sequence - 3 \n Super Giants - 4 | Hyper Giants - 5"

@app.route("/history")
def predict_history():
    output = ""
    for i in history:
        output += "values:" + str(i[0]) + "\n" + "prediction:" + str(i[1]) + "\n" + "Accuracy score:" + str(i[2]) + "\n\n"
    return output


app.run(port=8080, debug=True)