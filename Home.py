import streamlit as st
from pytrends.request import TrendReq
import pandas as pd
from textblob import TextBlob
import os

from llama_index import (
    GPTVectorStoreIndex, Document, SimpleDirectoryReader,
    QuestionAnswerPrompt, LLMPredictor, ServiceContext
)
import json
from langchain import OpenAI
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine



st.set_page_config(page_title=None, page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)

#image
st.markdown(
    """
    <style>
    .center {
        display: flex;
        justify-content: center;
        height:250px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Add the image with center alignment
st.markdown('<div class="center"><img src="data:image/jpeg;base64,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" alt="Image" width="250"></div>', unsafe_allow_html=True)

# page heading
st.markdown('<p style="font-size:40px; color:navy; text-align:center;">Understand the Galle Tourism Market with Guide_Bot</p> <br>', unsafe_allow_html=True)

# Initialize pytrends
pytrends = TrendReq(hl='en-US', tz=360)

# List of initial keywords
initial_keywords = ['Galle Tourism', 'Galle', 'Hotels Galle', 'Resorts Galle ','Koggala Beach Hotels', 'Unawatuna Beach Hotels ', 'Galle Restaurants', 'Bentota Hotels']

# Initialize session state


if 'data2' not in st.session_state:
    st.session_state['data2'] = pd.DataFrame()

if 'data3' not in st.session_state:
    st.session_state['data3'] = pd.DataFrame()

tab1, tab2, tab3, tab4 = st.tabs(["Search Query Data Analytics and Forecasting", "Sentimental Analysis", "Price Optimization", "Chatbot"])

with st.expander("data"):
    col1, col2, col3 =  st.columns(3)



#####################   tab1 #@######################################################

# with tab1:
    # Create a for keyword selection
selected_keywords = tab1.multiselect('Select existing keywords', initial_keywords)


# When keywords are selected, fetch data from Google Trends and display it
if tab1.button('Fetch Google Trends data for selected keywords'):
    # Define the payload
    kw_list = selected_keywords

    # Get Google Trends data
    pytrends.build_payload(kw_list, timeframe='today 5-y')

    # Get interest over time
    data = pytrends.interest_over_time()
    if not data.empty:
        data = data.drop(labels=['isPartial'],axis='columns')

        # Save the data to the session state
        if 'data' not in st.session_state:

# st.session_state['data'] = pd.DataFrame()
            st.session_state['data'] = data
if 'data' in st.session_state:
    col1.write("## Trends Data")

    col1.write(st.session_state['data'])







#####################   tab2 #@######################################################

# with tab2:
    # Upload file

uploaded_file = tab2.file_uploader("Upload scraped data for reviews")
if uploaded_file is not None:
    st.session_state['data2'] = pd.read_csv(uploaded_file)
    col2.write("## Sentimental Data")
    col2.write(st.session_state['data2'])
    



#####################   tab1 #@######################################################

# with tab3:
    # Upload file

uploaded_file2 = tab3.file_uploader("Upload scraped data for prices")
if uploaded_file2 is not None:
    st.session_state['data3'] = pd.read_csv(uploaded_file2)
    
    col3.write("## Pricing Data")
    
    col3.write(st.session_state['data3'])
    
  


# with tab4:

if tab4.button('Save data and create index'):
    # Check if the 'data' directory exists
    if not os.path.exists('data'):
        os.makedirs('data')

    # Save the data from session state to CSV files
    if 'data' in st.session_state:
        st.session_state['data'].to_csv('data/data.csv')
        st.success('Data saved successfully in data/data.csv')

    if not st.session_state['data2'].empty:
        st.session_state['data2'].to_csv('data/data2.csv')
        st.success('Data2 saved successfully in data/data2.csv')

    if not st.session_state['data3'].empty:
        st.session_state['data3'].to_csv('data/data3.csv')
        st.success('Data3 saved successfully in data/data3.csv')

    documents = SimpleDirectoryReader('data').load_data()
    index = GPTVectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    if "query_engine" not in st.session_state:
        st.session_state.query_engine = query_engine

# tab4.write("Chat Bot")

with tab4:
    st.markdown('<p style="font-size:37px; color:navy">Chat Bot</p>', unsafe_allow_html=True)

ques = tab4.text_input("Ask question")
ask = tab4.button("submit question")

if ask:
    response = st.session_state.query_engine.query(ques)
    st.write(response.response)
