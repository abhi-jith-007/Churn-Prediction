# coding: utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
from flask import Flask, request, render_template, jsonify
import pickle

app = Flask("__name__")

df_1 = pd.read_csv("first_telc.csv")

@app.route("/")
def loadPage():
    return render_template('home.html', query="")

@app.route("/", methods=['POST'])
def predict():
    # Retrieve inputs from the form
    inputQuery1 = request.form['query1']
    inputQuery2 = request.form['query2']
    inputQuery3 = request.form['query3']
    inputQuery4 = request.form['query4']
    inputQuery5 = request.form['query5']
    inputQuery6 = request.form['query6']
    inputQuery7 = request.form['query7']
    inputQuery8 = request.form['query8']
    inputQuery9 = request.form['query9']
    inputQuery10 = request.form['query10']
    inputQuery11 = request.form['query11']
    inputQuery12 = request.form['query12']
    inputQuery13 = request.form['query13']
    inputQuery14 = request.form['query14']
    inputQuery15 = request.form['query15']
    inputQuery16 = request.form['query16']
    inputQuery17 = request.form['query17']
    inputQuery18 = request.form['query18']
    inputQuery19 = request.form['query19']

    # Load the pre-trained model
    model = pickle.load(open("model.sav", "rb"))
    
    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7, 
             inputQuery8, inputQuery9, inputQuery10, inputQuery11, inputQuery12, inputQuery13, inputQuery14,
             inputQuery15, inputQuery16, inputQuery17, inputQuery18, inputQuery19]]
    
    new_df = pd.DataFrame(data, columns=[
        'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 'Partner', 'Dependents', 
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
        'PaperlessBilling', 'PaymentMethod', 'tenure'
    ])
    
    # Append new data to existing dataframe
    df_2 = pd.concat([df_1, new_df], ignore_index=True) 
    
    # Group the tenure into bins of 12 months
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    df_2['tenure_group'] = pd.cut(df_2.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)
    df_2.drop(columns=['tenure'], axis=1, inplace=True)   
    
    # Create one-hot encoded columns
    df_2 = pd.get_dummies(df_2, columns=[
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'tenure_group'
    ])
    
    # Ensure the feature names match the model
    model_features = model.feature_names_in_
    new_df = df_2[model_features]
    
    # Predict customer churn and obtain confidence score
    single = model.predict(new_df.tail(1))
    probability = model.predict_proba(new_df.tail(1))[:,1]
    p_val = probability[0] * 100  # convert to percentage

    # Determine suggestions based on churn risk and confidence percentage
    if single[0] == 1:
        output_text = "This customer is likely to churn!"
        confidence_text = "Confidence: {:.2f}%".format(p_val)
        if p_val < 10:
            suggestions = "Churn risk is very low. Continue monitoring the customer for any subtle changes."
        elif p_val < 20:
            suggestions = "Churn risk is low. Consider sending occasional engagement emails to keep them happy."
        elif p_val < 30:
            suggestions = "Churn risk is moderate. A small personalized discount or offer might help retain them."
        elif p_val < 40:
            suggestions = "Churn risk is noticeable. Improve customer support and address any minor issues promptly."
        elif p_val < 50:
            suggestions = "Churn risk is rising. Consider implementing a loyalty program and more frequent follow-ups."
        elif p_val < 60:
            suggestions = "Churn risk is moderate-high. Offer targeted promotions and personalized service enhancements."
        elif p_val < 70:
            suggestions = "Churn risk is high. Immediate outreach with special offers and personalized engagement is recommended."
        elif p_val < 80:
            suggestions = "Churn risk is very high. Aggressive retention strategies and proactive service improvements are needed."
        elif p_val < 90:
            suggestions = "Churn risk is extremely high. Urgent intervention is required with comprehensive customer support."
        else:
            suggestions = "Churn risk is critical. Immediate, intensive retention measures must be deployed to save the customer."
    else:
        output_text = "This customer is likely to continue!"
        confidence_text = "Confidence: {:.2f}%".format(p_val)
        if p_val < 10:
            suggestions = "Customer satisfaction is critical. Urgently deploy targeted initiatives to enhance their experience."
        elif p_val < 20:
            suggestions = "Customer satisfaction is low. Immediate measures, such as personalized engagement, are recommended."
        elif p_val < 30:
            suggestions = "Customer satisfaction is slipping. Identify pain points and act quickly with relevant incentives."
        elif p_val < 40:
            suggestions = "Customer satisfaction is borderline. Focus on tailored customer service to boost their experience."
        elif p_val < 50:
            suggestions = "Customer satisfaction is acceptable but could be improved. Consider proactive outreach to address any concerns."
        elif p_val < 60:
            suggestions = "Customer satisfaction is average. Engage them with personalized offers and gather feedback."
        elif p_val < 70:
            suggestions = "Customer satisfaction is solid. Small loyalty rewards and proactive support can further improve retention."
        elif p_val < 80:
            suggestions = "Customer satisfaction is good. Regular check-ins and minor incentives will be beneficial."
        elif p_val < 90:
            suggestions = "Customer satisfaction is high. Occasional appreciation emails can help maintain loyalty."
        else:
            suggestions = "Customer satisfaction is excellent. Keep up the current service quality and engagement."

    return render_template(
        'home.html', 
        output1=output_text, 
        output2=confidence_text, 
        suggestions=suggestions,
        query1=request.form['query1'], 
        query2=request.form['query2'],
        query3=request.form['query3'],
        query4=request.form['query4'],
        query5=request.form['query5'], 
        query6=request.form['query6'], 
        query7=request.form['query7'], 
        query8=request.form['query8'], 
        query9=request.form['query9'], 
        query10=request.form['query10'], 
        query11=request.form['query11'], 
        query12=request.form['query12'], 
        query13=request.form['query13'], 
        query14=request.form['query14'], 
        query15=request.form['query15'], 
        query16=request.form['query16'], 
        query17=request.form['query17'],
        query18=request.form['query18'],
        query19=request.form['query19']
    )

@app.route('/get-input-queries', methods=['GET'])
def takeCommand():
    try:
        input_queries = {}
        for i in range(1, 20):
            input_queries[f"inputquery{i}"] = input(f"Enter inputquery{i}: ")
        return input_queries
    except Exception as e:
        print("Some Error Occurred. Sorry from Luna")
        return jsonify({"error": "Some Error Occurred. Sorry from Luna"}), 500

if __name__ == '__main__':
    app.run(debug=True)
