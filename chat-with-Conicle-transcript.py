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
from PIL import Image
import base64
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

def get_image_as_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")
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

    system_instruction = """ 
    You are an AI generative chatbot designed to act as a mentor and coach, providing domain-specific expertise, support, and guidance. Users will select an AI agent specializing in a particular domain (such as soft skills, data science, etc.) and interact with you to achieve their goals.

Your primary tasks are to:

Understand the user's needs through active listening and targeted questions.
Help users set SMART (Specific, Measurable, Achievable, Relevant, Time-bound) goals.
Provide emotional support by acknowledging user emotions and offering encouragement.
Personalize responses based on the user's progress, preferences, and previous interactions.
You should first rely on the given knowledge base before using outside knowledge to answer the user's questions. Use outside knowledge for additional examples or support when it is advisable and enhances the user’s understanding without deviating from the core knowledge base.

Additionally, you should be able to detect when the conversation is ending by identifying cues such as the user's summary statements, declining number of questions, or direct indications. Suggest creating an assessment or quiz to help the user summarize their knowledge. Provide guidelines for assessments that are relevant to the user's goals and the specific domain.

Your answer should be in Thai."""

    model = GenerativeModel(model_name="gemini-1.5-flash",
                            system_instruction=system_instruction)



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
        page_title="Conicle Dos Chatbot",
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

    def get_category(category):
        st.session_state['category'] = category
        st.write(category)
        return category

    st.sidebar.button('Finance', on_click=get_category, args=('Finance',))
    st.sidebar.button('Data Science', on_click=get_category, args=('Data Science',))
    st.sidebar.button('ConicleSpace-Grow (BETA)', on_click=get_category, args=('ConicleSpace-Grow',))
    st.sidebar.button('Conicle Piece of Cake', on_click=get_category, args=('Piece_of_cake',))

    # Main content area for displaying chat messages
    # Load the image
    image_path = 'Sorc-Ai.png'
    img_base64 = get_image_as_base64(image_path)

    # Display the image next to the title
    st.markdown(
        f"""
            <div style="display: flex; align-items: center;">
                <img src="data:image/png;base64,{img_base64}" style="width:50px;height:50px;margin-right:10px;">
                <h1 style="display:inline;">AI Team Chatbot</h1>
            </div>
            """,
        unsafe_allow_html=True
    )
    st.write("อยากสอนมากๆค่ะ")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Chat input
    # Placeholder for chat messages

    if 'category' not in st.session_state:
        st.session_state['category'] = None

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
                category = st.session_state['category']
                response = user_input(user_question=prompt, category=category) #TODO Please specify the category here
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