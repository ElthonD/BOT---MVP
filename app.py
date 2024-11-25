####
#### Streamlit Streaming using LM Studio as OpenAI Standin
#### run with `streamlit run app.py`

# !pip install pypdf langchain langchain_openai 

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


# app config
path_img = './img/GIM Desarrollos Logo.png'

st.set_page_config(page_title="BOT Inteligente - MVP", page_icon="🤖")
col1, col2, col3 = st.columns([1,10,1])
with col2:
    st.title("BOT Inteligente - MVP")
st.image(path_img, use_container_width = True)

# Point to the local server
llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
#modelo = "local-model" # si tienes la API de pago: gpt-4 , gpt-3.5, etc

def get_response(user_query, chat_history):

    template = """
    Eres un asistente en español y ayudas respondiendo con la mayor exactitud posible:

    Chat history: {chat_history}

    User response: {user_question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    # Using LM Studio Local Inference Server
    #llm = ChatOpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
    #modelo = "local-model"
    chain = prompt | llm | StrOutputParser()
    
    return chain.stream({
        "chat_history": chat_history,
        "user_question": user_query,
    })

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hola, Soy el Asistente Virtual ACE. ¿Como puedo ayudarte?"),
    ]
    
# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("ACE"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Humano"):
            st.write(message.content)

# user input
user_query = st.chat_input("Escribe tu mensaje aquí...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Humano"):
        st.markdown(user_query)

    with st.chat_message("ACE"):
        response = st.write_stream(get_response(user_query, st.session_state.chat_history))

    st.session_state.chat_history.append(AIMessage(content=response))