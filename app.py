import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import streamlit as st
from tensorflow.keras.models import load_model


batting_teams = ['Kolkata Knight Riders', 'Mumbai Indians', 'Chennai Super Kings', 'Delhi Capitals', 
                 'Rajasthan Royals', 'Royal Challengers Bangalore', 'Punjab Kings', 'Sunrisers Hyderabad']
bowling_teams = ['Kolkata Knight Riders', 'Mumbai Indians', 'Chennai Super Kings', 'Delhi Capitals', 
                 'Rajasthan Royals', 'Royal Challengers Bangalore', 'Punjab Kings', 'Sunrisers Hyderabad']

batting_team_encoder = LabelEncoder()
batting_team_encoder.fit(batting_teams)

bowling_team_encoder = LabelEncoder()
bowling_team_encoder.fit(bowling_teams)

batter_encoder = LabelEncoder()
bowler_encoder = LabelEncoder()
non_striker_encoder = LabelEncoder()

batter_encoder.fit(['Player1', 'Player2', 'Player3']) 
bowler_encoder.fit(['PlayerA', 'PlayerB', 'PlayerC'])  
non_striker_encoder.fit(['PlayerX', 'PlayerY', 'PlayerZ'])

with open('batting_team.pkl', 'wb') as f:
    pickle.dump(batting_team_encoder, f)

with open('bowling_team.pkl', 'wb') as f:
    pickle.dump(bowling_team_encoder, f)

with open('batter.pkl', 'wb') as f:
    pickle.dump(batter_encoder, f)

with open('bowler.pkl', 'wb') as f:
    pickle.dump(bowler_encoder, f)

with open('non_striker.pkl', 'wb') as f:
    pickle.dump(non_striker_encoder, f)

def safe_transform(encoder, value):
    try:
        return encoder.transform([value])[0]
    except ValueError: 
        return encoder.transform([encoder.classes_[0]])[0]
    
def preprocess_input_data(batting_team, bowling_team, batter, bowler, non_striker, batsman_runs, extra_runs):
    with open('batting_team.pkl', 'rb') as f:
        batting_team_encoder = pickle.load(f)
    
    with open('bowling_team.pkl', 'rb') as f:
        bowling_team_encoder = pickle.load(f)
        
    with open('batter.pkl', 'rb') as f:
        batter_encoder = pickle.load(f)
        
    with open('bowler.pkl', 'rb') as f:
        bowler_encoder = pickle.load(f)
        
    with open('non_striker.pkl', 'rb') as f:
        non_striker_encoder = pickle.load(f)
    
    batting_team_encoded = safe_transform(batting_team_encoder, batting_team)
    bowling_team_encoded = safe_transform(bowling_team_encoder, bowling_team)
    batter_encoded = safe_transform(batter_encoder, batter)
    bowler_encoded = safe_transform(bowler_encoder, bowler)
    non_striker_encoded = safe_transform(non_striker_encoder, non_striker)

    input_data = pd.DataFrame([[batting_team_encoded, bowling_team_encoded, batter_encoded, bowler_encoded, non_striker_encoded, batsman_runs, extra_runs]], 
                              columns=['batting_team', 'bowling_team', 'batter', 'bowler', 'non_striker', 'batsman_runs', 'extra_runs'])
    
    input_data_reshaped = input_data.values.reshape((1, 1, input_data.shape[1]))
    
    return input_data_reshaped

model_lstm = load_model('lstm_model.h5')

st.title('IPL Prediction')

batting_team = st.selectbox('Select Batting Team', batting_teams)
bowling_team = st.selectbox('Select Bowling Team', bowling_teams)
batter = st.text_input('Enter Batsman Name')
bowler = st.text_input('Enter Bowler Name')
non_striker = st.text_input('Enter Non-Striker Name')
batsman_runs = st.number_input('Enter Batsman Runs', min_value=0)
extra_runs = st.number_input('Enter Extra Runs', min_value=0)

input_data = preprocess_input_data(batting_team, bowling_team, batter, bowler, non_striker, batsman_runs, extra_runs)

if st.button('Predict'):
    prediction = model_lstm.predict(input_data)
    predicted_runs = round(prediction[0][0])  
    st.write(f'Predicted Total Runs: {predicted_runs}')