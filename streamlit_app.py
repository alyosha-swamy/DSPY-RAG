# streamlit_app.py
import streamlit as st
from test import RAG, parse_questions  # Import the RAG class and other relevant functions from test.py

# Instantiate your RAG model
rag_model = RAG()

# Define the Streamlit app
def main():
    st.title("RAG System Hosted on Streamlit")
    
    # Text input for the user to ask a question
    question = st.text_input("Ask a question:")
    
    # Button to get the answer
    if st.button('Retrieve and Generate Answer'):
        if question:
            with st.spinner('Generating answer...'):
                # Use your RAG model to generate an answer
                answer = rag_model.forward(question)
                st.success('Done!')
                st.write(answer)
        else:
            st.warning('Please enter a question.')

if __name__ == "__main__":
    main()