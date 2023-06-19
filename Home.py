import streamlit as st
from pytrends.request import TrendReq
import pandas as pd
from textblob import TextBlob
st.set_page_config(page_title=None, page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)
# Initialize pytrends
pytrends = TrendReq(hl='en-US', tz=360)

# List of initial keywords
initial_keywords = ['Galle Tourism', 'Galle', 'Hotels Galle', 'Resorts Galle Srilanka','Srilanka', 'Tourist', 'locations']

# Initialize session state


if 'data2' not in st.session_state:
    st.session_state['data2'] = pd.DataFrame()

if 'data3' not in st.session_state:
    st.session_state['data3'] = pd.DataFrame()

tab1, tab2, tab3, tab4 = st.tabs(["Search Query Data Analytics and Forecasting", "Sentimental Analysis", "Price Optimization", "Chatbot"])

with st.expander("data"):
    col1, col2, col3 =  st.columns(3)



#####################   tab1 #@######################################################

with tab1.expander("Abirame"):
    st.write("#### Novel Contribution")
    st.write("The novel contribution of our research lies in the innovative integration of weather data with Google Trends information to account for seasonal variations. This unique amalgamation not only enhances the accuracy of our analytics by factoring in seasonal effects on trends, but also paves the way for more insightful understandings of how weather and trend patterns influence one another. Furthermore, we have optimized the use of Genetic Algorithms in our analysis. This optimization is facilitated through a groundbreaking approach that redefines the way these algorithms are traditionally used. This not only bolsters the robustness and efficiency of our algorithm but also ensures that it is better equipped to handle complex datasets and provide more precise insights. This novel approach to the aggregation of data and optimization of genetic algorithms positions our research at the cutting edge of data analytics and trend forecasting.")
    st.write("#### Implementation link")
    st.write("https://colab.research.google.com/drive/1erptkpWML8o3lWfxfQ1NQZw9hXNatrP8?usp=sharing")




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







#####################   tab1 #@######################################################

# with tab2:
    # Upload file

with tab1.expander("Sinthuja"):
    st.write("#### Novel Contribution")
    st.write("My study introduces a novel use of deep learning for the detection of sarcasm and sentiment analysis in reviews. We leverage a custom-designed architecture that integrates the BERT transformer-based deep learning model, which offers improved context understanding and semantic interpretation. This unique approach provides a more precise and nuanced text analysis, marking a significant advancement in the field..")
    st.write("#### Implementation link")
    st.write("https://colab.research.google.com/drive/1erptkpWML8o3lWfxfQ1NQZw9hXNatrP8?usp=sharing")


uploaded_file = tab2.file_uploader("Upload scraped data for reviews")
if uploaded_file is not None:
    st.session_state['data2'] = pd.read_csv(uploaded_file)
    col2.write("## Sentimental Data")
    col2.write(st.session_state['data2'])
    



#####################   tab1 #@######################################################

# with tab3:
    # Upload file

with tab3.expander("Gobinthiran"):
    st.write("#### Novel Contribution")
    st.write("My research presents a novel contribution to the field of price forecasting by leveraging the power of Deep Learning. The groundbreaking approach employs the Prophet model, a tool traditionally reserved for time series forecasting, but in this context, is used to predict price movements. However, what truly sets this research apart is the subsequent optimization of these forecasts using a custom architecture based on Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) models. This unique blend of advanced methodologies provides a new perspective on price forecasting, opening doors for further innovative applications and research in the financial and economic sectors.")
    st.write("#### Implementation link")
    st.write("https://colab.research.google.com/drive/1erptkpWML8o3lWfxfQ1NQZw9hXNatrP8?usp=sharing")


uploaded_file2 = tab3.file_uploader("Upload scraped data for prices")
if uploaded_file2 is not None:
    st.session_state['data3'] = pd.read_csv(uploaded_file2)
    
    col3.write("## Pricing Data")
    
    col3.write(st.session_state['data3'])
    
  


# with tab4:

with tab4.expander("Nikalya"):
    st.write("#### Novel Contribution")
    st.write("In my research, I present a novel contribution to the field of conversational AI by integrating Langchain-based Llama Index Chatbots into my methodology. These chatbots, powered by a unique combination of language models and indexing technology, provide significant advancements in the context-awareness and personalized responses of conversational agents. Moreover, my approach employs deep learning strategies, specifically leveraging transfer learning techniques, to produce dynamic and context-specific prompts. This allows our chatbots to excel in the retrieval of question-answer pairs, thus leading to more meaningful and informative interactions with users. The integration of these innovative techniques sets a new benchmark in chatbot technology, underscoring the potential of merging established deep learning approaches with emerging technologies like Langchain.")
    st.write("#### Implementation link")
    st.write("https://colab.research.google.com/drive/1erptkpWML8o3lWfxfQ1NQZw9hXNatrP8?usp=sharing")


tab4.write("Chat Bot")
tab4.text_input("Ask question")
tab4.button("submit answer")

