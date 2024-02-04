import streamlit as st
import pandas as pd 
import numpy as np
import requests
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Function to download the model file.
def download_model():
    model_url = "https://github.com/TolaniSilas/genre_prediction/blob/main/best_xgb_model.joblib"
    response = request.get(model_url)

    with open("best_xgb_model.joblib", 'wb') as file_ojbect:
        file_object.write(response.content)

# Download the model file by calling the function.
download_model()

# Load the model using joblib
loaded_model = joblib.load("best_xgb_model.joblib")





# Set the theme configuration.
st.set_page_config(
    page_title="Genre Prediction App",
    page_icon=":musical_note",
    layout="wide",
    initial_sidebar_state="expanded"
)


app_mode = st.sidebar.selectbox("Select Page", ["Home", "Deep Dive into Music Analysis",  "Prediction"])

# # Display if app_mode is "Home".
# if app_mode == 'Home':
    
#     st.title("Genre Prediction App")
#     st.markdown("This is a web app that predicts a song as either Hiphop or Rock")
#     st.image("genre.png")

#     st.markdown("Datasets:")
#     datasets = pd.read_csv("genre_dataset.csv")
#     datasets = datasets.drop("Unnamed: 0", axis=1)
#     st.write(datasets.head())
#     st.write("Developed by: Osunba Silas")
    
    



# Your Streamlit code
if app_mode == 'Home':
    st.title("Genre Prediction App")
    st.markdown("This is a web app that predicts a song as either Hiphop or Rock")
    st.image("genre.png")

    st.markdown("Datasets:")
    datasets = pd.read_csv("genre_dataset.csv")
    datasets = datasets.drop("Unnamed: 0", axis=1)
    st.write(datasets.head())
    # st.write("Developed by: Osunba Silas")
    
    # HTML code with embedded Streamlit content and styling
    html_code = f"""
    <div style="padding: 10px; background-color: #18578A; margin-bottom: 10px;">
        <h2 style="color: white;">About the App</h2>
        <p style="color: white;">A Genre Prediction App is an application that uses machine learning algorithms and music analysis techniques to predict
        the genre of a given song. It analyzes various features of a song, such as tempo, instrumentation and vocal characteristics, and then applies a
        trained model to classify the song into Hip Hop and Rock genres.</p>
    
        <h3 style="color: white;">Datasets:</h3>
        
        <!-- Your "Developed by" text here -->
    </div>
    
    <div style="background-color: blue; padding: 10px;">
        
    </div>
    """

    # Display the HTML content using st.markdown
    st.markdown(html_code, unsafe_allow_html=True)










    
    
    
    
    
# Display if app_mode is "Prediction".   
elif app_mode == "Prediction":
    st.header("Prediction")
    st.markdown("To predict if a song is Hip Hop or Rock, fill in the following:")
    
    # Prompt the user input features so as to predict if it is a Hip Hop or Rock song.
    acousticness = st.number_input("Acousticness", min_value=0.0, max_value=1.0)
    danceability = st.number_input("Danceability", min_value=0.0, max_value=1.0)
    energy = st.number_input("Energy", min_value=0.0, max_value=1.0)
    instrumentalness = st.number_input("Instrumentalness", min_value=0.0, max_value=1.0)
    liveness = st.number_input("Liveness", min_value=0.0, max_value=1.0)
    speechiness = st.number_input("Speechiness", min_value=0.0, max_value=1.0)
    tempo = st.number_input("Tempo", min_value=0.0, max_value=260.0)
    valence = st.number_input("Valence", min_value=0.0, max_value=1.0)
    
    
    
    if st.button("Predict the song genre", key="my_button"):
        # Prepare the user input as an array.
        user_input = [float(acousticness), float(danceability), float(energy), float(instrumentalness), \
                      float(liveness), float(speechiness), float(tempo), float(valence)]
        
        # Perform prediction using the loaded model.
        prediction = loaded_model.predict([user_input])

        # Display the prediction.
        st.success(f"The song genre is {prediction[0]}")
           
    
# Display if app_mode is "Deep Dive"      
elif app_mode == "Deep Dive into Music Analysis":
    st.header("Deep Dive into Music Analysis")
    st.markdown("In this section, we'll explore the fascinating word of music analysis and its impact on understanding different genres.")
    
    acousticness = """Acousticness: The acousticness of a music refers to the degree of acoustic elements in a song. It measures how much the 
    sound of the music is derived from acoustic instruments or natural sounds, as opposed to electronic or synthesized sounds. The value of 
    acousticness can range from 0.0 to 1.0. A value of 0.0 indicates that the music has no acoustic elements and is completely electronic or 
    synthesized, while a value of 1.0 indicates that the music is entirely acoustic, with no electronic or synthesized elements. The values in
    between represent varying degrees of acousticness in the music."""
    st.write(acousticness)
    
    danceability = """Danceability: Danceability is a measure of how suitable a song is for dancing. It can range from 0.0 to 1.0, with 0.0 meaning 
    the song is not at all danceable and 1.0 indicating that the song has a high potential for dancing. Values in between represent varying degrees of 
    danceability. So, the closer the value is to 1.0, the more danceable the song is considered to be."""
    st.write(danceability)
    
    energy = """Energy: Energy is a measure of the intensity and activity level in a song. It can range from 0.0 to 1.0, with 0.0 representing a low-
    energy song that is calm and mellow, and 1.0 representing a high-energy song that is energetic and lively. The values in between indicate varying 
    degrees of energy in the music. So, the closer the value is to 1.0, the more energetic and lively the song is considered to be."""
    st.write(energy)
    
    instrumentalness = """Instrumentalness: Instrumentalness is a measure of the presence of vocals in a song. It can range from 0.0 to 1.0. A value 
    of 0.0 indicates that the song is likely to have vocals, while a value of 1.0 suggests that the song is likely to be instrumental without any 
    vocals. The values in between represent varying degrees of the presence of vocals in the music. So, the closer the value is to 1.0, the more 
    likely it is that the song is instrumental."""
    st.write(instrumentalness)
    
    liveness = """Liveness: Liveness is a measure of the presence of a live audience or performance in a song. It can range from 0.0 to 1.0. A value of 0.0 
    indicates that the song is likely a studio recording without any live elements, while a value of 1.0 suggests that the song is a live performance 
    with an audible audience. The values in between represent varying degrees of live elements in the music. So, the closer the value is to 1.0, 
    the more likely it is that the song contains a live performance or audience sounds."""
    st.write(liveness)
    
    speechiness = """Speechiness: Speechiness is a measure of the presence of spoken words in a song. It can range from 0.0 to 1.0. A value of 0.0 indicates that
    the song is mostly instrumental, while a value of 1.0 suggests that the song is primarily spoken word or a podcast. The values in between represent 
    varying degrees of the presence of spoken words in the music. So, the closer the value is to 1.0, the more likely it is that the song contains 
    spoken words."""
    st.write(speechiness)
    
    tempo = """Tempo: Tempo is a measure of the speed or pace of a song. It is typically measured in beats per minute (BPM). The range of tempos in 
    music can vary greatly, but generally, tempos can range from as low as 40 BPM to as high as 200+ BPM. Slower tempos are often associated with more
    relaxed or calm music, while faster tempos are often associated with energetic or upbeat music. So, depending on the song, the tempo can fall within
    a wide range of values."""
    st.write(tempo)
    
    valence = """Valence: Valence is a measure of the musical positivity or negativity of a song. It can range from 0.0 to 1.0. A valence value of 
    0.0 represents a song with a negative or sad emotional tone, while a valence value of 1.0 represents a song with a positive or happy emotional tone. 
    The values in between represent varying degrees of emotional positivity or negativity in the music. So, the closer the value is to 1.0, the more
    positive and uplifting the song is considered to be."""
    st.write(valence)
    
    
    
    
    
    
    
