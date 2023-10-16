import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import plotly.express as px
from streamlit_ace import st_ace
from joblib import load
import random
import pickle
import requests
from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup as bs
import math
from torchtext.vocab import GloVe

st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout='wide')

# Create a sidebar and populate with content
st.sidebar.title("About 📰")
st.sidebar.info(
    """
    This is a website that detects whether a site contains fake news or genuine information!
    """
)
st.sidebar.info(
    """
    Made by: Vidhi Patra
    """
)
st.title("FakeFinder ✅")

st.header("How AI Can Help Detect Fake News?")

st.markdown('---')

st.markdown(
    '''
    By training a machine learning model with a diverse range of fake and real news articles, 
    we empower ourselves to preemptively filter out fake content on platforms, preventing 
    them from gaining enough traction to significantly sway public opinion.

Explore the capabilities of my fake news detection tool by entering any website and receiving our AI's assessment of the article.
    '''
)

# AI Fake News Demo
st.subheader(" 🗞️ Try the AI Powered Fake News Detector 📰 ")

model = pickle.load(open('fake_news_model.sav', 'rb'))

news_site = st.text_input(
    '''
    Paste a link in the space: 
    '''
)

VEC_SIZE = 300

glove = GloVe(name='6B', dim=VEC_SIZE)

def get_word_vector(word):
    try:
      return glove.vectors[glove.stoi[word.lower()]].numpy()
    except KeyError:
      return None

def get_data_pair(url):
    if not url.startswith('http'):
        url = 'http://' + url
    url_pretty = url
    if url_pretty.startswith('https://'):
        url_pretty = url_pretty[7:]
    if url_pretty.startswith('https://'):
        url_pretty = url_pretty[8:]
    response = requests.get(url, timeout=10)
    htmltext = response.text

    return url_pretty, htmltext

def dict_to_features(features_dict):
  X = np.array(list(features_dict.values())).astype('float')
  X = X[np.newaxis, :]
  return X

def get_normalized_count(html, phrase):
    return math.log(1 + html.count(phrase.lower()))

def keyword_featurizer(url, html):
    features = {}
    
    # Same as before.
    features['.com domain'] = url.endswith('.com')
    features['.org domain'] = url.endswith('.org')
    features['.net domain'] = url.endswith('.net')
    features['.info domain'] = url.endswith('.info')
    features['.org domain'] = url.endswith('.org')
    features['.biz domain'] = url.endswith('.biz')
    features['.ru domain'] = url.endswith('.ru')
    features['.co.uk domain'] = url.endswith('.co.uk')
    features['.co domain'] = url.endswith('.co')
    features['.tv domain'] = url.endswith('.tv')
    features['.news domain'] = url.endswith('.news')
    
    keywords = ['trump', 'biden', 'clinton', 'sports', 'finance']
    
    for keyword in keywords:
      features[keyword + ' keyword'] = get_normalized_count(html, keyword)
    
    return features

def glove_transform_data_descriptions(descriptions):
    X = np.zeros((len(descriptions), VEC_SIZE))
    for i, description in enumerate(descriptions):
        found_words = 0.0
        description = description.strip()
        for word in description.split(): 
            vec = get_word_vector(word)
            if vec is not None:
                found_words += 1
                X[i] += vec
        if found_words > 0:
            X[i] /= found_words        
    return X

vectorizer = CountVectorizer(max_features=300)

train_descriptions = pickle.load(open('train_descriptions', 'rb'))

vectorizer.fit(train_descriptions)

def vectorize_data_descriptions(descriptions, vectorizer):
  X = vectorizer.transform(descriptions).todense()
  return X

def get_description_from_html(html):
  soup = bs(html)
  description_tag = soup.find('meta', attrs={'name':'og:description'}) or soup.find('meta', attrs={'property':'description'}) or soup.find('meta', attrs={'name':'description'})
  if description_tag:
    description = description_tag.get('content') or ''
  else: # If there is no description, return empty string.
    description = ''
  return description

def combine_features(X_list):
  return np.concatenate(X_list, axis=1)

def featurize_data_pair(url, html):
  # Approach 1.
  keyword_X = dict_to_features(keyword_featurizer(url, html))
  # Approach 2.
  description = get_description_from_html(html)
  
  bow_X = vectorize_data_descriptions([description], vectorizer)
  
  # Approach 3.
  glove_X = glove_transform_data_descriptions([description])
  
  X = combine_features([keyword_X, bow_X, glove_X])
  
  return X

clicked = st.button("Generate Prediction")

if clicked:
    url, html = get_data_pair(news_site)
    curr_X = featurize_data_pair(url, html)
    curr_y = model.predict(curr_X)[0]
    if curr_y < 0.5:
        st.balloons()
        st.write("You submitted: " + news_site)
        st.subheader("🥳 The AI predicts that this site is real news! 🦾")
    else:
        st.write("You submitted: " + news_site)
        st.subheader("🤡 The AI predicts that this site is fake news! 👻")

