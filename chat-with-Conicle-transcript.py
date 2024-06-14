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
import pandas as pd

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


def get_conversational_chain(prompt, mode):
    vertexai.init(project='conicle-ai', credentials=credentials)

    if mode == "Coach":
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

    if mode == "Personalized Learning Advisor":
        system_instruction = """
                You are an AI generative chatbot designed to help users build their own learning paths and personalities by selecting the content they want to learn. Users will select content categories and specific courses, and you will provide personalized insights based on their selections.

                Your primary tasks are to:

                Understand the user's learning preferences through their selected content.
                Provide comprehensive analysis of the user's learning personality and behavior.
                Suggest recommended learning strategies, future learning paths, and potential career paths.
                Personalize responses based on the user's progress, preferences, and previous interactions.

                Your answer should be in Thai."""
    model = GenerativeModel(model_name="gemini-1.5-flash",
                            system_instruction=system_instruction)



    response = model.generate_content(
        [prompt]
    )

    print(response.text)
    return response.text


def user_input(user_question, category=None, mode ="Coach"):
    if mode == "Coach" and category:
        vector_store = create_vector_database(category)
        doc = vector_store.similarity_search(user_question, k=6)
        prompt = f"""Context:\n {doc}?\n Question: \n{user_question}\n"""
    else:
        chat_history = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages if msg['role'] != "system"])
        prompt = f"Chat History:\n{chat_history}\nUser: {user_question}\nAssistant:"

    response = get_conversational_chain(prompt, mode)
    return response


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "เริ่มแชทกับ Conicle AI ได้เลย!"}]
    st.session_state['category'] = None
    st.session_state['initial_analysis_done'] = False



def main():
    st.set_page_config(
        page_title="Conicle Dos Chatbot",
        page_icon="🤖"
    )

    categories = {
        "Data Science": [
            "Introduction to Data Science",
            "Advanced Data Science Techniques",
            "Data Analysis with Python",
            "Machine Learning with R",
            "Big Data Analytics"
        ],
        "Cloud Computing": [
            "Cloud Basics",
            "Advanced Cloud Architecture",
            "AWS Certified Solutions Architect",
            "Azure Fundamentals",
            "Google Cloud Platform for Developers"
        ],
        "Cybersecurity": [
            "Network Security Essentials",
            "Ethical Hacking",
            "Cybersecurity for Beginners",
            "Advanced Cyber Threats",
            "Certified Information Systems Security Professional (CISSP)"
        ],
        "AI": [
            "Introduction to AI",
            "Reinforcement Learning",
            "Natural Language Processing",
            "AI for Healthcare",
            "AI Ethics and Policy"
        ],
        "Machine Learning": [
            "Machine Learning Basics",
            "Advanced Machine Learning",
            "Deep Learning with TensorFlow",
            "Computer Vision",
            "Unsupervised Learning Techniques"
        ],
        "Software Development": [
            "Introduction to Software Development",
            "Software Development Best Practices",
            "Full-Stack Web Development",
            "Agile Project Management",
            "DevOps Essentials"
        ],
        "Product Management": [
            "Introduction to Product Management",
            "Agile Product Management",
            "Product Lifecycle Management",
            "Market Research for Product Managers",
            "Product Strategy and Roadmapping"
        ],
        "Finance": [
            "Introduction to Finance",
            "Corporate Finance",
            "Financial Markets and Instruments",
            "Investment Banking",
            "Fintech and Innovation"
        ],
        "Marketing": [
            "Introduction to Marketing",
            "Digital Marketing",
            "Content Marketing Strategy",
            "SEO and SEM",
            "Social Media Marketing"
        ],
        "Human Resources": [
            "Introduction to Human Resources",
            "Recruitment and Talent Acquisition",
            "Employee Relations",
            "HR Analytics",
            "Compensation and Benefits"
        ],
        "Healthcare": [
            "Introduction to Healthcare Management",
            "Healthcare Data Analytics",
            "Public Health",
            "Healthcare Policy and Economics",
            "Clinical Research and Trials"
        ]
    }
    # Initialize session state
    if 'user_choices' not in st.session_state:
        st.session_state.user_choices = {}

    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "เริ่มแชทกับ น้อง Brae ได้เลย!"}
        ]

    if 'initial_analysis_done' not in st.session_state:
        st.session_state.initial_analysis_done = False
    if 'mode' not in st.session_state:
        st.session_state['mode'] = None
    #overall_content = pd.read_csv("transcripts/Contentniverse.csv", usecols=['Course Name', 'Category'])
    recommend_categories = {
        "Data Science": [
            "Introduction to Data Science",
            "Advanced Data Science Techniques",
            "Data Analysis with Python",
            "Machine Learning with R",
            "Big Data Analytics"
        ],
        "Cloud Computing": [
            "Cloud Basics",
            "Advanced Cloud Architecture",
            "AWS Certified Solutions Architect",
            "Azure Fundamentals",
            "Google Cloud Platform for Developers"
        ]
    }

    recommended_df = pd.DataFrame(recommend_categories.items(), columns = ['category', 'course_name'])
    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Menu:")

        # Mode selection
        mode = st.radio(
            "Select AI Mode",
            options=["Starter", "Coach", "Personalized Learning Advisor"],
            index=0 if st.session_state['mode'] is None else ["Starter","Coach", "Personalized Learning Advisor"].index(
                st.session_state['mode'])
        )
        st.session_state['mode'] = mode

        if mode == "Personalized Learning Advisor":
            st.write("Select the categories and courses you are interested in:")
            for category, courses in categories.items():
                with st.expander(category):
                    selected_courses = st.multiselect(f"Select courses in {category}:", courses, key=category)
                    if selected_courses:
                        st.session_state.user_choices[category] = selected_courses

            if st.button("วิเคราะห์ฉันสิ"):
                # Combine all selected courses into a single string
                all_choices = []
                for cat, courses in st.session_state.user_choices.items():
                    for course in courses:
                        all_choices.append(f"{course} in {cat}")
                choices_str = "; ".join(all_choices)
                prompt = f"""
    The user has selected the following courses and categories:
    {choices_str}

    Based on these selections, please provide a comprehensive analysis of the user's learning personality and behavior, including:

    1. Overall learning preferences and tendencies
    2. Recommended learning strategies
    3. Suggested future learning paths and resources
    4. Potential career paths or roles

    You have access to a DataFrame of courses with the following schema: course_name, category. Use this DataFrame to recommend contents that align with the user's personality. If no suitable course is found, kindly respond with: "ขออภัย, จากบุคลิกภาพของคุณ, ฉันไม่สามารถแนะนำเนื้อหาที่มีอยู่ได้" (Sorry, based on your personality, I can't suggest existing contents).

    5. Recommend suitable contents: Based on the DataFrame provided, suggest relevant courses and encourage the user to explore these contents on 'Coniverse' in a friendly and motivating tone. Start with an introduction like "จากสิ่งที่เรามีอยู่, เราแนะนำว่า..." (Based on what we have, we suggest...).

    Your answer should be in Thai.

    {recommended_df}
    """

                response = get_conversational_chain(prompt, mode)  # Pass the selected mode
                st.session_state.user_choices['response'] = response
                st.session_state.initial_analysis_done = True

                # Append AI response to chat messages

    def get_category(category):
        st.session_state['category'] = category
        st.write(category)
        return category

    if st.sidebar.button("ใช้ยังไงใช่ป่ะ"):
        st.sidebar.markdown("""
                            ## How to Use This Project (โปรเจคนี้ทำมาเพื่อการทดลอง experiment, fine-tuning system instruction เพื่อนำไปใช้ใน AI สืบไป)
                            0. เลือกหัวข้อด้านซ้ายมือ
                            1. แล้วคุย-ปรึกษา-ถามได้เลย
                            2. **Clear Chat History**: Use the clear chat history button to reset the chat. (Reset บทสนทนา)

                            ### Features
                            - **Category Selection**: Filter documents by category.
                            - **AI Chatbot**: Interact with the AI for guidance and support.
                            - **Clear Chat History**: Reset the chat for a new session.
                        """)

    if mode == "Coach":
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


    # Display the image next to the title
    st.markdown(
        f"""
            <div style="display: flex; align-items: center;">
                <img src="data:image/png;base64,{img_base64}" style="width:50px;height:50px;margin-right:10px;">
                <h1 style="display:inline;">AI Team Lab</h1>
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

    if not st.session_state.initial_analysis_done:
        if prompt := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

        # Display chat messages and bot response
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    if st.session_state['category'] is not None:
                        category = st.session_state['category']
                        response = user_input(user_question=prompt, category=category, mode=mode)
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

        # Display the AI response if it exists
    if 'response' in st.session_state.user_choices:
        with st.chat_message("assistant"):
            st.write(st.session_state.user_choices['response'])

if __name__ == "__main__":
    main()