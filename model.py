# -*- coding: utf-8 -*-
"""
Created on Tue May 18 00:32:18 2021

@author: VijayKrishna
"""

import pandas as pd
import numpy as np
import sklearn
import nltk
import re 
import requests
import contractions
import pickle
import warnings
warnings.filterwarnings("ignore")
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer() 
from nltk.stem.snowball import SnowballStemmer
sn = SnowballStemmer(language='english')

from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split



######################################################################### Pre - Process ####################################################

def remove_columns(df,l):
    return df.drop(l,axis=1)

def convert_str(df):
    return  df.Review.apply(lambda x: str(x))  # to string

def convert_lower(s):
    return s.apply(lambda x: x.lower()) # Lowering the text


def remove_url(s):
    return s.apply(lambda x: re.sub(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))",'',x))


def remove_repeated(k):
    return k.apply(lambda x: re.sub(r"(.)\1{2,}",r"\1" ,x))  # hi!!!!! --> hi! 

def remove_contractions(k):
    return k.apply(lambda x: contractions.fix(x)) # don't - do not 



def remove_special(u):
    return u.apply(lambda x: re.sub(r"[^a-zA-Z\s]+",'',x))
      
def lemmet(l):                                                                                                     # StopWords Removal
    return l.apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in nltk.word_tokenize(x) if not word in set(stopwords.words('english')) ]))
        
def stem(c):
    return c.apply(lambda x: " ".join([sn.stem(word) for word in nltk.word_tokenize(x)]))


def preprocess(df):
  
    df.Review = stem(lemmet(remove_contractions(remove_special(remove_repeated(remove_url(convert_lower(convert_str(df))))))))

    return df

#####################################################################Sentiment-Analysis###################################################
# Importing Dataset

df = pd.read_csv('sample30.csv')

df.dropna(subset=['user_sentiment'],axis=0,inplace=True)


# Data PreProcessing

df_1 = df[['reviews_title','reviews_text','user_sentiment']]

df_1['reviews_title'].fillna("",inplace=True) # Nan --> Blank Spaces
df_1['Review']=df_1['reviews_title']+" "+df_1['reviews_text']
df_1.drop(['reviews_title','reviews_text'],axis=1,inplace=True)
df_1 = df_1[['Review','user_sentiment']]
df_1['user_sentiment'] =df_1['user_sentiment'].map({"Positive":1,"Negative":0})
df_clean = preprocess(df_1)

# Mapping File 

df_2 = df[['name','reviews_title','reviews_text']]
df_2['reviews_title'].fillna("",inplace=True) # Nan --> Blank Spaces
df_2['Review']=df_2['reviews_title']+" "+df_2['reviews_text']
df_2.drop(['reviews_title','reviews_text'],axis=1,inplace=True)
df_main_cleaned= preprocess(df_2)
#df_main_cleaned.to_csv('df_mapping.csv',header=True,index=False)


# Modelling

X = df_clean['Review']
y = df_clean['user_sentiment']


# Vectorizing

vectorizer = TfidfVectorizer(use_idf = True,strip_accents='ascii')
X_vec = vectorizer.fit_transform(X)


#filename = 'vectorizer.pk'
#pickle.dump(vectorizer, open(filename, 'wb'))


#Spilitting

X_train,X_test,y_train,y_test = train_test_split(X_vec,y,
                                                    train_size=0.7,
                                                    test_size = 0.3, random_state=47)


X_train = X_train.toarray()

X_test = X_test.toarray()



# Training

clf_logreg = LogisticRegression(C=2, class_weight='balanced', dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=47, solver='liblinear', tol=0.0001, verbose=0,
                   warm_start=False)


clf_logreg.fit(X_train,y_train)

# Saving the Model
#filename = 'logreg.sav'
#pickle.dump(clf_logreg, open(filename, 'wb'))

#################################################################### User-User Recommendation###############################################

df_2 = df[['name','reviews_username','reviews_rating']].dropna(axis=0)


# Splitting
df_train,df_test = train_test_split(df_2,
                                                    train_size=0.7,
                                                    test_size = 0.3, random_state=47)



# Dummy Train and Dummy Test
dummy_train = df_train.copy()


# Assuming Products not rated by user are not purchased by them and assigning 1 
dummy_train['reviews_rating'] = dummy_train['reviews_rating'].apply(lambda x: 0 if  x>=0 else 1) 


# pivoting the dummy train 
df_pivot_dummy = dummy_train.pivot_table(index ='reviews_username',columns ='name',values ='reviews_rating').fillna(1)
df_pivot_dummy.head()


# Create a user-product matrix.
df_pivot = df_train.pivot_table(
    index ='reviews_username',columns ='name',values ='reviews_rating')


# Normalizing ratings
mean = np.nanmean(df_pivot, axis=1)
df_subtracted = (df_pivot.T-mean).T


# Creating the User Similarity Matrix using pairwise_distance function.
user_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
user_correlation[np.isnan(user_correlation)] = 0


# Prediction 
user_correlation[user_correlation<0]=0

# Dot Product
user_predicted_ratings = np.dot(user_correlation, df_pivot.fillna(0))

user_final_rating = np.multiply(user_predicted_ratings,df_pivot_dummy)
print('Done!')

#user_final_rating.to_csv('user_final_rating.csv')

#Recco = user_final_rating.loc['00sab00'].sort_values(ascending=False)[0:20]