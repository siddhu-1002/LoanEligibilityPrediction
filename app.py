from flask import Flask, render_template, request
import pickle
import joblib
import pandas as pd



# Loading the encoders and scaler
# enc_gender = pickle.load(open("enc_gender.obj", "rb"))   ----- male - 1, female - 0
# enc_married = pickle.load(open("enc_married.obj", "rb")) ----- no - 0, yes - 1
# enc_emp = pickle.load(open("enc_emp.obj", "rb")) ----- no - 0, yes - 1


le_edu = pickle.load(open("le_edu.obj", "rb"))
le_prop = pickle.load(open("le_prop.obj", "rb"))
scaler = joblib.load("std_scaler.bin")

# Loading the prediction model
model = pickle.load(open('model.pkl', 'rb'))


# names of the dataframe columns
cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
       'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area']


# create flask app
app = Flask(__name__)

@app.route("/home", methods = ["GET"])
def home():
    return render_template("home.html")

@app.route("/response", methods = ["POST"])
def response():


    gender = request.form["gender"]
    if gender == "Male":
        gender = 1.0
    else:
        gender = 0.0

    married = request.form["married"]
    if married == "Yes":
        married = 1.
    else:
        married = 0.

    dependents = int(request.form["dep"])
    education = request.form["edu"]

    self_employed = request.form["emp"]
    if self_employed == "Yes":
        self_employed = 1.
    else:
        self_employed = 0.

    app_inc = int(request.form["app"])
    coapp_inc = int(request.form["coapp"])
    loan_amount = int(request.form["loan_amt"])
    loan_amt_term = int(request.form["lat"])
    credit_history = float(request.form["cred"])
    property_area = request.form["prop"]

    ls = [[gender, married, dependents, education, self_employed, app_inc, coapp_inc,
    loan_amount, loan_amt_term, credit_history, property_area]]
    
    new_df = pd.DataFrame(ls, columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
       'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area'])

    # The columns that need to be encoded
    # one_hot_columns = ["Gender", "Married", "Self_Employed"]
    # label_encode_columns = ["Education", "Property_Area"]


    # print(new_df["Gender"].shape)
    # print(enc_married.categories_)
    # print(enc_emp.categories_)


    # # one hot encoding the categorical variables
    # new_df["Gender"] = enc_gender.transform(new_df["Gender"])
    # new_df["Married"] = enc_married.transform(new_df["Married"])
    # new_df["Self_Employed"] = enc_emp.transform(new_df["Self_Employed"])

    # # # label encoding the varibles
    new_df["Education"] = le_edu.transform(new_df["Education"])
    new_df["Property_Area"] = le_prop.transform(new_df["Property_Area"])
    
    new_df = scaler.transform(new_df)
    
    value = model.predict(new_df)

    if value[0] == 1:
        value = "Congratulations!"
        return render_template("success.html", value = value)
    else:
        value = "Oops!"
        return render_template("failure.html", value = value)


    



# Entry point of the flask application
if __name__ == "__main__":
    # 5000 is default port on localhost
    app.run(debug = True)