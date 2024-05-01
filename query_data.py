import os
import speech_recognition as sr
import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain.vectorstores.chroma import Chroma
from langchain_cohere import CohereEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatHuggingFace
from langchain.prompts import ChatPromptTemplate
from gtts import gTTS
import time

CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def check_db_prompt(query_text):
    # Prepare the DB.
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_function = CohereEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=10)
    if not results:
        return "Unable to find matching results."

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = ChatHuggingFace(llm=HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
    ))

    response_text = model.predict(prompt)
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"{query_text}\n\nResponse: {response_text}\nSources: {sources}"
    response_text = response_text.strip().replace("`", "").replace("\n", " ")
    index = response_text.find("<|assistant|>")
    if index != -1:
        response_text = response_text[index + len("<|assistant|>"):]
    return response_text

def voice_chat_interface():
    st.title("Voice Chat Interface")
    

    r = sr.Recognizer()
    chat_container = st.container()
    should_listen = False

    def on_button_clicked():
        nonlocal should_listen
        should_listen = True

    if st.button("Ask Question", on_click=on_button_clicked):
        with sr.Microphone() as source:
            st.subheader("Listening Question...")
            print("Listening...")
            audio = r.listen(source)

            try:
                query = r.recognize_google(audio, language='en-in')
                st.write("User: ", query)

                response = check_db_prompt(query)
                st.write("Assistant:\n", response)

                # Text-to-speech conversion
                tts = gTTS(text=response, lang="en", slow=False)
                tts.save("response.mp3")
                st.audio("response.mp3")
                os.remove("response.mp3")

            except Exception as e:
                print("I could not recognize your Query: ", e)

if __name__ == "__main__":
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    os.environ["COHERE_API_KEY"] = "RZFvd31jwHnH0RVw317LnrNumRtkVZBGV9D5W0bp"
    voice_chat_interface()

