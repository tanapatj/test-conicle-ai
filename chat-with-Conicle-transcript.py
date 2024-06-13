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
            print(file)
            print(category.lower())
            if category.lower() in file.lower():
                print("correct")
                print(file)
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
    You are an AI generative chatbot designed to act as a mentor and coach, providing domain-specific expertise, support, and guidance for a specific course. Users will select an AI agent specializing in a particular domain (such as soft skills, data science, etc.) and interact with you to enhance their learning experience and achieve their goals within this course.

Your primary tasks are to:

1.Understand the user's needs through active listening and targeted questions related to the course content.
2.Help users set SMART (Specific, Measurable, Achievable, Relevant, Time-bound) goals within the context of the course.
3.Provide emotional support by acknowledging user emotions and offering encouragement throughout their learning journey.
4.Personalize responses based on the user's progress, preferences, and previous interactions specific to the course material.
5.Answer questions related to the course, provide suggestions for additional resources, and help users navigate through the course content effectively.
6.Use the provided course knowledge base as the primary source of information, resorting to outside knowledge only when it enhances understanding without deviating from the core knowledge base.
7.Detect when the conversation is ending by identifying cues such as the user's summary statements, declining number of questions, or direct indications. Suggest creating an assessment or quiz to help the user summarize their knowledge. Provide guidelines for assessments that are relevant to the user's goals and the specific domain.
Your responses should be in Thai, using language and tone appropriate for a coaching environment."""

    model = GenerativeModel(model_name="gemini-1.5-flash",
                            system_instruction=system_instruction)



    response = model.generate_content(
        [prompt]
    )

    print(response.text)
    return response.text


def user_input(user_question, category=None):
    vector_store = create_vector_database(category)
    doc = vector_store.similarity_search(user_question, k=6)
    prompt = f"""Context:\n {doc}?\n Question: \n{user_question}\n"""

    response = get_conversational_chain(prompt)
    return response


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "เริ่มแชทกับ Conicle AI ได้เลย!"}]
    st.session_state['category'] = None


def main():
    st.set_page_config(
        page_title="Conicle Dos Chatbot",
        page_icon="🤖"
    )

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Menu:")

    def get_category(category):
        st.session_state['category'] = category
        st.write(category)
        return category

    st.sidebar.button('ConicleX Course IC Plain-Paper 1 Exam Preparation: Make It Easy with Mind Map ตอนที่ 2', on_click=get_category, args=('Finance',))
    st.sidebar.button('ConicleX Course Mastering Prompt Engineering Design for ChatGPT AI Part 2', on_click=get_category, args=('Data Science',))
    #st.sidebar.button('ConicleSpace-Grow (BETA)', on_click=get_category, args=('ConicleSpace-Grow',))
    #st.sidebar.button('Conicle Piece of Cake', on_click=get_category, args=('Piece_of_cake',))
    st.sidebar.button('ConicleX Course Cybersecurity Awareness', on_click=get_category, args=('course_123',))
    st.sidebar.button('ConicleX The Mindset Makeover', on_click=get_category, args=('course_124',))
    st.sidebar.button('ConicleX How to Increase Your Confidence', on_click=get_category, args=('course_125',))
    st.sidebar.button('ConicleX Piece of Cake Good Communication', on_click=get_category, args=('course_126',))
    st.sidebar.button('ConicleX Piece of Cake Happy Workplace', on_click=get_category, args=('course_127',))
    st.sidebar.button('ConicleX Piece of Cake ISO', on_click=get_category, args=('course_128',))
    st.sidebar.button('ConicleX Piece of Cake Strategic Thinking', on_click=get_category, args=('course_129',))
    st.sidebar.button('ConicleX Piece of Cake คู่มือผู้จัดการพันธุ์ใหม่', on_click=get_category, args=('course_130',))
    st.sidebar.button('ConicleX Piece of Cake เพิ่มยอดขาย', on_click=get_category, args=('course_131',))

    # Main content area for displaying chat messages
    # Load the image
    image_path = 'Sorc-Ai.png'
    img_base64 = get_image_as_base64(image_path)

    if st.button("ใช้ยังไง"):
        st.markdown("""
                            ## How to Use This Project (โปรเจคนี้ทำมาเพื่อการทดลอง experiment, fine-tuning system instruction เพื่อนำไปใช้ใน AI Coaching)
                            0. **Select AI Mode**: Coaching / Learning Path Builder
                            1. **Select Category**: Choose the appropriate category you want to master. (เลือกหัวข้อที่ต้องการจะถูก Coach)
                            2. **Chat**: Use the chat input to interact with the AI. (..)
                            3. **Clear Chat History**: Use the clear chat history button to reset the chat. (Reset บทสนทนา)

                            ### Features
                            - **Category Selection**: Filter documents by category.
                            - **AI Chatbot**: Interact with the AI for guidance and support.
                            - **Clear Chat History**: Reset the chat for a new session.
                        """)
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
                if(st.session_state['category'] is not None):
                    category = st.session_state['category']
                    response = user_input(user_question=prompt, category=category) #TODO Please specify the category here
                else:
                    response = "เลือกหัวข้อที่จะถามก่อน"
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