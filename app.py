import openai
import streamlit as st
import pandas as pd
import os
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI, OpenAI

#set the page title
st.set_page_config(page_title="Where are we going?")

#set the openai api key
openai.api_key = st.secrets["OPENAI_API_KEY"]

def main():
    
    #set the page header
    st.header("Where would you like to stay tonight?")
    
    #prompt the user to ask a question
    user_question = st.text_input("Where would you like to stay tonight?")

    #create CSV agent using the langchain python package
    agent = create_csv_agent(
            ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
            "data/usaPlaces2yr.csv",
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
        )
    
    #run the agent based on user question
    if user_question is not None and user_question != "":
        result = agent.invoke(user_question)
        #display the response
        st.write(result)

    #allow the user to have a conversation with the chatbot
    st.text_area("Conversation", height=200)

if __name__ == "__main__":
    main()
