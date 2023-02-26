from flask import Flask,render_template,request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
with open('F://New folder (2)//model.pkl', 'rb') as file:
    model = pickle.load(file)
car=pd.read_csv('F://New folder (2)//Cleaned_data.csv')

@app.route('/',methods=['GET','POST'])

def index():
    Km_driven = car['km_driven']
    Fuel = car['fuel']
    Seller_type = car['seller_type']
    Transmission = car['transmission']
    Owner = car['owner']
    Seats = car['seats']
    Torque = car['torque_rpm']
    Mileage = car['Mileage']
    Engine = car['Engine']
    Max_power = car['Max_Power']
    return render_template('index.html',Km_driven=Km_driven,Fuel=Fuel,Seller_type=Seller_type,Transmission=Transmission,Owner=Owner,Seats=Seats,Torque=Torque,Mileage=Mileage,Engine=Engine,Max_power=Max_power)

@app.route('/predict',methods=['POST'])
def predict():

    Km_driven=request.form.get('Km_driven')
    Fuel=request.form.get('Fuel')
    Seller_type=request.form.get('Seller_type')
    Transmission=request.form.get('Transmission')
    Owner=request.form.get('Owner')
    Seats=request.form.get('Seats')
    Torque=request.form.get('Torque')
    Mileage=request.form.get('Mileage')
    Engine=request.form.get('Engine')
    Max_power=request.form.get('Max_power')

    prediction= model.predict(pd.DataFrame(columns=['km_driven','fuel','seller_type','transmission','owner','seats','torque_rpm','Mileage','Engine','Max_Power'],
                              data=np.array([Km_driven,Fuel,Seller_type,Transmission,Owner,Seats,Torque,Mileage,Engine,Max_power]).reshape(1, 10)))
    print(prediction)

    return str(np.round(prediction[0],2))
if __name__=='__main__':
    app.run(debug=True)