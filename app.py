from flask import Flask,render_template,request
import pandas as pd 
import pickle
import numpy as np


app=Flask(__name__)
data=pd.read_csv('cleaned_data.csv')
pipe=pickle.load(open('ridgeModel.pkl','rb'))

@app.route('/')
def index():
    locations=sorted(data['location'].unique())
    return render_template("index.html",locations=locations )

@app.route('/predict',methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')

    # Check if values are not empty
    if bhk and bath and sqft:
        bhk = int(bhk)
        bath = int(bath)
        sqft = float(sqft)

        input_data = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
        prediction = pipe.predict(input_data)[0] * 1e5
        return str(np.round(prediction, 2))
    else:
        return "Please provide valid input values."

if __name__=="__main__":
    app.run(debug=True,port=5001)