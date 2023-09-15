!pip install spacy
import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import warnings
import spacy
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from sklearn.feature_extraction.text import TfidfVectorizer

def load_model(modelfile):
	loaded_model = pickle.load(open(modelfile, 'rb'))
	return loaded_model

def get_key(val,my_dict):
	for key,value in my_dict.items():
		if val == value:
			return key

model = load_model('model.pkl')

vec_model = load_model('vectorizer.pkl')

def main():
    st.beta_set_page_config(page_title="Medical Symptoms Text Classification", page_icon="💊", layout='centered', initial_sidebar_state='auto')
    # title
    html_temp = """
    <div>
    <h1 style="color:STEELBLUE;text-align:left;">Health ScanAI 🩺 </h1>
    </div>
    """
    
    st.markdown(html_temp, unsafe_allow_html=True)
    
    
    '''
    ## How does it work ❓ 
    เขียนอาการของโรคที่เกิดขึ้นในกล่องข้อความ
    '''
    
    '''
    #### วันนี้รู้สึกอย่างไรบ้าง ? 
    '''
    med_text = st.text_area("", "Write Here")
    prediction_labels = {'Emotional pain': 0, 'Hair falling out':1, 'Head hurts':2, 'Infected wound':3, 'Foot achne':4,
    'Shoulder pain':5, 'Injury from sports':6, 'Skin issue':7, 'Stomach ache':8, 'Knee pain':9, 'Joint pain':10, 'Hard to breath':11,
    'Head ache':12, 'Body feels weak':13, 'Feeling dizzy':14, 'Back pain':15, 'Open wound':16, 'Internal pain':17, 'Blurry vision':18,
    'Acne':19, 'Neck pain':21, 'Cough':22, 'Ear achne':23, 'Feeling cold':24}
    
    if st.button("Classify"):
        vec_text =  vec_model.transform([med_text]).toarray()
        pred = model.predict(vec_text)
        final_result = get_key(pred,prediction_labels)
        st.warning((final_result))

    
    hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    </style>
    """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

if __name__ == '__main__':
	main()
