from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def main():
    load_dotenv()
    st.set_page_config(page_title="🦜🔗 Chat with Document")
    st.header("🦜🔗 Chat with Document")
    
    # upload file
    pdf = st.file_uploader("Upload your document", type="pdf")
    
    # extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
            
        # split into chuncks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        chunks = text_splitter.split_text(text)
        # st.write(chunks)
        
        # create embeddings
        
        with get_openai_callback() as cb:
            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)

            user_question = st.text_input("Ask your question about PDF:")

            chain = load_qa_chain(OpenAI(model_name="gpt-3.5-turbo"), 
                      chain_type="stuff") # we are going to stuff all the docs in at once
            if user_question:
                docs = knowledge_base.similarity_search(user_question)
                response = chain.run(input_documents=docs, question=user_question)
                st.write(response)
                with st.expander('OpenAI Usage'):
                    st.info(cb)
        
        

if __name__ == '__main__':
    main()