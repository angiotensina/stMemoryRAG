import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS, Pinecone
from langchain.chat_models import ChatOpenAI
# from langchain.models import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
# from langchain.llms import HuggingFaceHub
import pinecone
from langchain.llms import OpenAI

# Configura tu clave API de OpenAI (reemplaza 'tu_clave_api' con tu clave API real)
api_key_gpt4 = os.environ.get("OPENAI_API_KEY")

# Configuración del modelo GPT-4 de 32K, si está disponible
# Reemplaza 'gpt-4-32k-model-name' con el nombre correcto del modelo, si existe
model_name = "gpt-4"

pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment=os.environ.get("PINECONE_API_ENV")
)
index = pinecone.Index('cdc-doc1')

try:
    indexl = pinecone.list_indexes()
except Exception as e:
    print(f"Error al listar los índices de Pinecone: {e}")


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    """
    load_dotenv(find_dotenv(), override=True)
    api_key_pinecone = os.environ.get("PINECONE_API_KEY")
    api_env_pinecone = os.environ.get("PINECONE_API_ENV")
    # print(api_key_pinecone)
    # print(api_env_pinecone)

    pinecone.init(api_key=api_key_pinecone, environment=api_env_pinecone)
    """
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    # vectorstore2 = pinecone.from_texts(texts=text_chunks, embedding=embeddings)
    # vectorstore = Pinecone.from_texts()

    return vectorstore

    """
    def get_vector_pinecone(txt_chunks):
        #vectorpinecone = Pinecone

        #return txt_chunks
    """


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput():
    
    response = st.session_state.conversation({'question': st.session_state.user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    """
    # Crear una instancia del modelo GPT-4 utilizando LangChain
    gpt4 = OpenAI(
        api_key="OPENAI_API_KEY", model=model_name)

    # Interactuar con el modelo
    prompt = "Escribe una explicación sobre la teoría de la relatividad de Einstein."
    response = gpt4.generate(prompt, max_tokens=100)

    # Imprimir la respuesta
    st.write(response)

    # INICIALIZANDO PINECONE
    pinecone.init(
        api_key=os.environ.get("PINECONE_API_KEY"),
        environment=os.environ.get("PINECONE_API_ENV")
    )
    index = pinecone.Index('cdc-doc1')
    """

    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    # st.write(index)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    st.text_input("Ask a question about your documents:", on_change=handle_userinput, key="user_question")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)
                print(vectorstore)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()


