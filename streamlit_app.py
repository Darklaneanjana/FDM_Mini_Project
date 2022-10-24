#hello world streamlit app

import streamlit as st
import numpy as np
import pandas as pd
from tensorflow import keras
from PIL import Image
from joblib import load
pd.set_option("display.precision", 2)

movies = pd.read_csv('dataset/movies.csv')
avgRatingDF = pd.read_csv('avgRatingDF.csv')
movies = movies.join(avgRatingDF.set_index("movieId"), on="movieId")
item_train1 = pd.read_csv("movieVector.csv").astype('float64').drop(columns=['(no genres listed)'])
categories = np.loadtxt('categories.csv', delimiter=',', dtype='str')
user_train2 = pd.read_csv("userVector.csv").astype('float64')

model = keras.models.load_model('my_model')
scalerUser=load('scalerUser.bin')
scalerItem=load('scalerItem.bin')
scalerTarget=load('scalerTarget.bin')


def newUserPredict(user_vec):
       # generate and replicate the user vector to match the number movies in the data set.
       user_train1 = pd.DataFrame(user_vec, index=[0])
       user_train1 = pd.DataFrame(np.repeat(user_train1.values, item_train1.shape[0], axis=0), columns=user_train1.columns) 

       # scale our user and item vectors
       suser_vecs = scalerUser.transform(user_train1.to_numpy())
       sitem_vecs = scalerItem.transform(item_train1.to_numpy())

       # make a prediction
       y_p = model.predict([suser_vecs[:, 3:], sitem_vecs[:, 1:]])

       # unscale y prediction 
       y_pu = scalerTarget.inverse_transform(y_p)

       # sort the results, highest prediction first
       sorted_index = np.argsort(-y_pu,axis=0).reshape(-1).tolist()  #negate to get largest rating first
       sorted_ypu   = y_pu[sorted_index]
       sorted_items = movies.iloc[sorted_index]  #using unscaled vectors for display
       # sorted_items['rating'] = sorted_ypu
       sorted_items.insert(1, 'y_predict', sorted_ypu)

       #displplay sorted_items dataframe in streamlit
       st.write(sorted_items.iloc[:10])



def existingUserPredict(uid):
       st.write(f'#### User Id: {uid}')
       user_train2 = pd.read_csv("userVector.csv").astype('float64')
       user_train2 = user_train2.loc[user_train2['userId'] == uid]
       user_train2 = pd.DataFrame(np.repeat(user_train2.values, item_train1.shape[0], axis=0), columns=user_train2.columns) 
       user_train2.head()

       # form a set of user vectors. This is the same vector, transformed and repeated.
       item_vecs = item_train1.to_numpy()
       user_vecs = user_train2.to_numpy()

       # scale our user and item vectors
       suser_vecs = scalerUser.transform(user_vecs)
       sitem_vecs = scalerItem.transform(item_vecs)

       # # make a prediction
       y_p = model.predict([suser_vecs[:, 3:], sitem_vecs[:, 1:]])

       # # unscale y prediction 
       y_pu = scalerTarget.inverse_transform(y_p)

       # sort the results, highest prediction first
       sorted_index = np.argsort(-y_pu,axis=0).reshape(-1).tolist()  #negate to get largest rating first
       sorted_ypu   = y_pu[sorted_index]
       sorted_items = movies.iloc[sorted_index]  #using unscaled vectors for display
       sorted_items.insert(1, 'y_predict', sorted_ypu)
       # sorted_user  = user_train2.iloc[sorted_index]
       # sorted_y     = y_vecs[sorted_index]

       genres = sorted_items['genres'].str.split('|', expand=False)
       user_genre_ave = []
       try:
              for i in range(len(genres)):
                     if genres[i] != ['(no genres listed)']:
                            user_genre_ave.append(user_train2.iloc[uid][genres[i]].values.round(1)) 
                     else:
                            user_genre_ave.append(0)
       except Exception as e:
              print(genres[i])
              print(e)

       sorted_items.insert(2, 'genre', user_genre_ave)

       st.write(sorted_items.iloc[:10])


st.write('# Content Based Movie Recommender System')
st.write('## Predictions for a new user')
options = st.multiselect('Select Your Favourite ',categories,['Action', 'Adventure'])
# st.write('You selected:', options)

newButton = st.button('Predict_for_new_user')
user_vec = {'userId':0, 'userRatingCount':0, 'userAvgRating':0, 'Action':0, 'Adventure':0,
              'Animation':0, 'Children':0, 'Comedy':0, 'Crime':0, 'Documentary':0, 'Drama':0,
              'Fantasy':0, 'Film-Noir':0, 'Horror':0, 'IMAX':0, 'Musical':0, 'Mystery':0,
              'Romance':0, 'Sci-Fi':0, 'Thriller':0, 'War':0, 'Western':0}

if newButton:
       for i in options:
              user_vec[i] = 5
       newUserPredict(user_vec)


st.write('## Predictions for an existing user')

uid = 1
st.text_input('Enter User Id', 1,key="placeholder")
uid = int(st.session_state.placeholder)
st.write(user_train2.loc[user_train2['userId'] == uid])
existingButton = st.button('Predict_for_existing_user')
if existingButton:
       existingUserPredict(uid)

htp5= 'http://img.omdbapi.com/?apikey=a50d9a01&i=tt1201607'
st.image(htp5, caption= '80-day sale data', width=100)
