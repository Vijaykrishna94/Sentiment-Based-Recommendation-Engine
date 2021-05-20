import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    user_input = [str(x) for x  in request.form.values()][0]
    try:
        
 # Reading the User-User Pivoted File
        user_final_rating = pd.read_csv('user_final_rating.csv')
        user_final_rating = user_final_rating.set_index('reviews_username')
        df = user_final_rating.loc[user_input].sort_values(ascending=False)[0:20]
    
 # Reading the Model File
        infile = open('logreg.sav','rb')
        logreg = pickle.load(infile)
        infile.close()

 # Reading the Mapping file 
        df_main = pd.read_csv('df_mapping.csv')
  
 # Reading the Vectorizer
        infile = open('vectorizer.pk','rb')
        vec = pickle.load(infile)
        infile.close()
    
    
    
        k=[]
        for i in df.index:
            k.append(logreg.predict(vec.transform(df_main[df_main.name==i]['Review'])).mean())
        df = df.reset_index().rename(columns={'index':'Product'})
        df['Sentiment'] = round(pd.Series(k),2)
        
    
        output = df.sort_values(by=['Sentiment'],ascending=False)[['Product','Sentiment']][0:5]
        final_output = render_template('index.html',table=output.to_html(index=False,justify='center',bold_rows = True))
    except:
        
        final_output =render_template('index.html',prediction_text = 'Invalid User Name) 
    return final_output



if __name__ == "__main__":
    app.run(debug=True)
