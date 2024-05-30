import os
import streamlit as st
import glob
import google.generativeai as genai
import lancedb
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import LanceDB
from io import StringIO
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

def create_vector_database(category=None):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")
    db = lancedb.connect("/tmp/lancedb")
    table = db.create_table(
        "my_table",
        data=[
            {
                "vector": embeddings.embed_query("Hello World"),
                "text": "Hello World",
                "id": "1",
            }
        ],
        mode="overwrite",
    )

    # Load the document, split it into chunks, embed each chunk and load it into the vector store.
    doc_list = []
    dl_dir = 'transcripts/'
    for file in glob.glob(dl_dir + "/*.txt"):
        if category is not None:
            print("CATEGORY CASE")
            if category.lower() in file.lower():
                with open(file) as f:
                    doc_list.append(f.read())
        else:
            print("ALL CATEGORY CASE")
            with open(file) as f:
                doc_list.append(f.read())

    text_splitter = CharacterTextSplitter(separator=',', chunk_size=100000, chunk_overlap=1000)
    documents = text_splitter.create_documents(doc_list)
    print('doclist', doc_list)
    vector_store = LanceDB.from_documents(documents, embeddings, connection=table)

    return vector_store


def get_conversational_chain(prompt):
    vertexai.init(project='conicle-ai', credentials=credentials)

    model = GenerativeModel(model_name="gemini-1.5-flash",
                            system_instruction="You are an AI generative chatbot designed to act as a friendly and knowledgeable coach and mentor. Your primary goal is to provide helpful and accurate answers to users' questions while fostering a supportive and engaging conversation. You should encourage users to explore their thoughts and feelings, offering both practical advice and emotional support")

    response = model.generate_content(
        [prompt]
    )

    print(response.text)
    return response.text


def user_input(user_question, category=None):
    vector_store = create_vector_database(category)
    doc = vector_store.similarity_search(user_question, k=4)
    prompt = f"""Context:\n {doc}?\n Question: \n{user_question}\n"""

    response = get_conversational_chain(prompt)
    return response


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "เริ่มแชทกับ Conicle AI ได้เลย!"}]


def main():
    st.set_page_config(
        page_title="Conicle Punny Chatbot",
        page_icon="🤖"
    )

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Menu:")
        # txt_file = st.file_uploader(
        #     "Upload your file(s) and Click on the Submit & Process Button", accept_multiple_files=False)
        if st.button("ประมวลผล!"):
            with st.spinner("Processing..."):
                # stringio = StringIO(txt_file.getvalue().decode("utf-8"))
                # raw_text = stringio.read()

                # raw_text = ""
                # for file in glob.glob(dl_dir + "/*.txt"):
                #     print(file)
                #
                #     my_file = open(file)
                #     raw_text += my_file.read()
                # get_conversational_chain(prompt=)
                # get_vector_store(text_chunks)
                create_vector_database(category='Finance')
                st.success("Done")

    # Main content area for displaying chat messages
    st.title("Punny AI Chatbot")
    st.write("อยากคุย!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Chat input
    # Placeholder for chat messages

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "เริ่มแชทกับ Conicle AI ได้เลย!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(user_question=prompt, category='Finance') #TODO Please specify the category here
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)

if __name__ == "__main__":
    main()