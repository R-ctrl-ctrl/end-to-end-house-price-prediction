from flask import Flask,render_template,request
import pandas as pd
import pickle
pipe = pickle.load(open('pipe.pkl','rb'))
df = pd.read_csv('cleaned_data.csv')
locations = df.location.unique()

app = Flask(__name__)
pred_x = " "

@app.route('/',methods=['GET','POST'])
def hello():
    if(request.method=='POST'):
        location = request.form['locations']
        bhk = request.form['bhk']
        bath = request.form['bathrooms']
        sqft = request.form['sqft']
        input = pd.DataFrame([[location,float(bhk),sqft,float(bath),]]  ,columns=['location','size','total_sqft','bath'])
        pred = pipe.predict(input)
        return render_template('index.html',pred=pred[0],locations=locations)
    return render_template('index.html',locations=locations)


# @app.route('/predict',methods=['POST'])
# def predict():
#     location = request.form['locations']
#     bhk = request.form['bhk']
#     bath = request.form['bathrooms']
#     sqft = request.form['sqft']
#     print('fetching')
#     print(location,bath,bhk,sqft)
#     return ""
    

if __name__ == '__main__':
    app.run(debug=True)