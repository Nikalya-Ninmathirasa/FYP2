import streamlit as st
from pytrends.request import TrendReq
import pandas as pd
from textblob import TextBlob

# Initialize pytrends
pytrends = TrendReq(hl='en-US', tz=360)

# List of initial keywords
initial_keywords = ['Galle Tourism', 'Galle', 'Hotels Galle', 'Resorts Galle Srilanka','Srilanka', 'Tourist', 'locations']

# Initialize session state
if 'data' not in st.session_state:
    st.session_state['data'] = pd.DataFrame()

tab1, tab2, tab3, tab4 = st.tabs(["Search Query Data Analytics and Forecasting", "Sentimental Analysis", "Price Optimization", "Chatbot"])

with tab1:
    # Create a for keyword selection
    selected_keywords = st.multiselect('Select existing keywords', initial_keywords)

    # Allow additional keywords to be added
    additional_keyword = st.text_input("Add a new keyword")
    if additional_keyword:
        selected_keywords.append(additional_keyword)

    # When keywords are selected, fetch data from Google Trends and display it
    if st.button('Fetch Google Trends data for selected keywords'):
        # Define the payload
        kw_list = selected_keywords

        # Get Google Trends data
        pytrends.build_payload(kw_list, timeframe='today 5-y')

        # Get interest over time
        data = pytrends.interest_over_time()
        if not data.empty:
            data = data.drop(labels=['isPartial'],axis='columns')

            # Save the data to the session state
            st.session_state['data'] = data

            st.write(data)

with tab2:
    # Assuming that sentiments are calculated on a 'text' column in the dataframe
    # st.session_state['data']['sentiment'] = st.session_state['data']['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    st.write(st.session_state['data'])

with tab3:
    # Dummy example of adding a price column
    # st.session_state['data']['price'] = st.session_state['data']['sentiment'] * 100  # Modify this as per your logic
    st.write(st.session_state['data'])

with tab4:
    st.write(st.session_state['data'])
