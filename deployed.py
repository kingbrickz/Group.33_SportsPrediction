import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

sc=StandardScaler()
model='"C:\Users\Maxwell\OneDrive - Ashesi University\Desktop\group 33\best22_dt (1).pk1"'
loaded_model = pickle.load(open(model, 'rb'))


st.title('My Streamlit Player Overall Prediction App')
potential = st.slider('Potential', 0, 100)
passing= st.slider('Passing', 0, 100)
wage_eur = st.slider('Wage', 0, 1000000)
value_eur= st.slider('Physicality', 0, 1000000000)
movement_reaction = st.slider('Movement Reaction', 0, 100)
power_shot_power= st.slider('Man Marking', 0, 100)
dribbling = st.slider('Dribbling', 0, 100)
passing = st.slider('Passing', 0, 100)
release_cluase_eur = st.slider('Release clause', 0, 100)
mentality_composure = st.slider('Mentality Composure', 0, 100)



def predict( 'movement_reactions', 'passing', 'mentality_composure',
       'dribbling', 'potential', 'wage_eur', 'power_shot_power', 'value_eur',
       'release_clause_eur'):
    X_pred = np.array([ 'movement_reactions', 'passing', 'mentality_composure',
       'dribbling', 'potential', 'wage_eur', 'power_shot_power', 'value_eur',
       'release_clause_eur'])
    X_pred_scaled = sc.fit_transform(X_pred.reshape(1, -1))
    rating = loaded_model.predict(X_pred_scaled)
    return rating



if st.button('Predict'):
    # Perform prediction using the loaded model
    prediction = loaded_model.predict([[ 'movement_reactions', 'passing', 'mentality_composure',
       'dribbling', 'potential', 'wage_eur', 'power_shot_power', 'value_eur',
       'release_clause_eur']])  
    st.write(f'Prediction: ', prediction)