from flask import Flask,render_template,request,url_for,jsonify

import pickle
import numpy as np
app = Flask(__name__)


f1 = open('enc_iris.pkl','rb')
f3 = open('clf_iris.pkl','rb')

enc = pickle.load(f1)

clf = pickle.load(f3)

@app.route("/<nums>",methods=['GET'])
def index(nums):
    nums=nums.split(',')
    d=dict()
    for key,num in enumerate(nums):
        d[key] = float(num)
    arr = np.array([nums])
    pred = clf.predict(arr)
    p=enc.inverse_transform(pred)[0].upper()
    
    return jsonify({"pred": p})

# @app.route("/predict",methods=['POST','GET'])
# def predict():
#     vals = [float(x) for x in request.form.values()]
#     arr = np.array([vals])
#     pred = clf.predict(arr)

   

#     return render_template('index.html',preds=enc.inverse_transform(pred)[0].upper(),vals=vals,pred=pred)

if __name__ == "__main__":
    app.run(debug=True)
