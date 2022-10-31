from altair.vegalite.v4.schema import Color
import streamlit as st
from streamlit_option_menu import option_menu
import requests

import numpy as np
import pandas as pd
from tensorflow import keras
from joblib import load
pd.set_option("display.precision", 2)

#loading the datasets
movies = pd.read_csv('dataset/movies.csv')
avgRatingDF = pd.read_csv('data/avgRatingDF.csv')
movies = movies.join(avgRatingDF.set_index("movieId"), on="movieId")
item_train1 = pd.read_csv("data/movieVector.csv").astype('float64').drop(columns=['(no genres listed)'])
categories = np.loadtxt('data/categories.csv', delimiter=',', dtype='str')
user_train2 = pd.read_csv("data/userVector.csv").astype('float64')
links = pd.read_csv('dataset/links.csv')

#loading the model and scalers
model = keras.models.load_model('data/my_model')
scalerUser=load('data/scalerUser.bin')
scalerItem=load('data/scalerItem.bin')
scalerTarget=load('data/scalerTarget.bin')

# funciton to display the movie row for new users
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

       return(sorted_items.iloc[:100])

# function to display the movie row for existing users
def existingUserPredict(uid):
       user_train2 = pd.read_csv("data/userVector.csv").astype('float64')
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
                     if genres.iloc[i] != ['(no genres listed)']:
                            user_genre_ave.append(user_train2.iloc[uid][genres.iloc[i]].values.round(1)) 
                     else:
                            user_genre_ave.append(0)
       except Exception as e:
              print(genres[i])
              print(e)
       sorted_items.insert(2, 'genre', user_genre_ave)

       return(sorted_items.iloc[:100])

# Title and sidebar paddings
st.markdown("""
       <style>div.block-container{padding-top:2rem;max-width :950px}</style>
       <style>.css-128j0gw {margin-top: -40px;}</style>
       """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
       selected = option_menu(
              menu_title='Select the user type',
              options = ['New User', 'Existing Users'],
              default_index=0,
              icons=['person', 'people'],
              menu_icon='bars',
       )
       st.write('''
              <br>
              <p style = "color:LightSeaGreen;" >
              The model is a deep learning model that uses a neural network to predict the rating of a movie for a user. The model is trained on the <a href="https://grouplens.org/datasets/movielens/">MovieLens</a> dataset. The dataset contains 100,836 ratings and 3,706 tag applications applied to 9,742 movies by 610 users. The dataset was generated on October 17, 2019. It was created by GroupLens Research at the University of Minnesota.</p>
                 
              ## How to use the app
              - Select the user type
              - For new users, enter the user vector
              - For existing users, select the user id
              - Click on the predict button
              - The top 10 movies will be displayed

              <br>
              <p style = "color:LightSeaGreen;" >
              The images and other details are fetched from <a href="https://www.imdb.com/">Internet Movie Database</a></p>
       ''', unsafe_allow_html=True)

# Display Movie List
def displayMovies(movies,j=0):
       for i in range(len(movies)):
                            with st.container():
                                   row = movies.iloc[i].values.tolist()
                                   imdbID = links.loc[links['movieId'] == row[0]].values.tolist()[0][1]
                                   imdbID = str(int(imdbID)).zfill(7)
                                   response = requests.get(imdbLink+imdbID).json()
                                   col1, col2, col3 = st.columns([2,4,3])

                                   with col1:
                                          st.image(htp5+imdbID, width=150)
                                   with col2:
                                          st.write(f'''
                                          <h3>{row[2+j]}</h3>
                                                        {row[3+j]}<br>
                                                        Rating:  {round(row[4+j],1)}
                                                 ''', unsafe_allow_html=True)
                                   with col3:
                                          st.markdown(f'''
                                                 <p style = "color:LightSeaGreen;" >
                                                 {response.get('Plot')}
                                                 <br><br>
                                                 Actors: {response.get('Actors')}
                                                 </p>
                                                 ''', unsafe_allow_html=True)
                            st.markdown('<hr>', unsafe_allow_html=True)
       
# Display user Vector
def displayUserRow(uid):   
       rowe = user_train2.loc[user_train2['userId'] == uid].iloc[:,1:]
       # rowe = round(rowe,1)
       hide_table_row_index = """
              <style>
              thead tr th:first-child {display:none}
              tbody th {display:none}
              </style>
            """
       # Inject CSS with Markdown
       st.markdown(hide_table_row_index, unsafe_allow_html=True)
       st.text('Selected user\'s Genre preferences')
       st.table(rowe.style.format("{:.1f}"))

# Main
st.write('<h1 style = "font-family:serif;font-size:46px;" >Content Based Movie Recommender System</h1>', unsafe_allow_html=True)
st.write('<br>' , unsafe_allow_html=True)

htp5= 'https://img.omdbapi.com/?apikey=a50d9a01&i=tt'
imdbLink = 'http://www.omdbapi.com/?apikey=a50d9a01&i=tt'
user_vec = {'userId':0, 'userRatingCount':0, 'userAvgRating':0, 'Action':0, 'Adventure':0,
                     'Animation':0, 'Children':0, 'Comedy':0, 'Crime':0, 'Documentary':0, 'Drama':0,
                     'Fantasy':0, 'Film-Noir':0, 'Horror':0, 'IMAX':0, 'Musical':0, 'Mystery':0,
                     'Romance':0, 'Sci-Fi':0, 'Thriller':0, 'War':0, 'Western':0}

# New User
if selected== 'New User':
       st.write('## Predictions for a new user')
       st.write('<br>', unsafe_allow_html=True)
       options = st.multiselect('Select Your Favourites',categories,['Action', 'Adventure'])
       # st.write('You selected:', options)

       col, buff2 = st.columns([2,4])
       # col.text_input('smaller text window:')
       end =   col.number_input('Enter Movies per predict' , min_value=10, max_value=len(movies), step=10)
       newButton = st.button('Predict for me')

       if newButton:
              with st.spinner('Beep Boop I\'m Thinking...'):
                     for i in options:
                            user_vec[i] = 5
                     movies = newUserPredict(user_vec)
                     st.markdown('<br>', unsafe_allow_html=True)
                     displayMovies(movies[0:end],0)


# Existing User
if selected== 'Existing Users':
       st.write('## Predictions for an existing user')

       # uid = 1
       uid = st.number_input('Enter User Id' , min_value=1, max_value=len(user_train2), step=1)
       uid = int(uid)
       displayUserRow(uid)
       

       col1, col2 = st.columns([2,4])
       # col.text_input('smaller text window:')
       end =   col2.number_input('Enter Movies per predict' , min_value=10, max_value=len(movies), step=10)
       agree = col1.checkbox('Show me the dataframe of movie predictions')
       st.markdown('<br>', unsafe_allow_html=True)
       existingButton = st.button('Predict for existing user')

       if existingButton:
              with st.spinner('Beep Boop I\'m Thinking...'):
                     try:
                            movies = existingUserPredict(uid)
                            if agree:
                                   st.write(movies)
                            st.write('<br>' , unsafe_allow_html=True)
                            displayMovies(movies[0:end],1)
                     except:
                            st.error('An Error Accured. Please check the user id')