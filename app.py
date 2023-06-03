from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
import pickle

def main():
    load_dotenv()
    st.set_page_config(page_title="🦜🔗 Chat with Document(s)")
    st.header("🦜🔗 Chat with Document(s)")
    
    # upload file
    pdf_list = st.file_uploader("Upload your document", type="pdf", accept_multiple_files=True)
    combined_text = ""
    
    # extract the text
    if pdf_list is not None:
        for pdf in pdf_list:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                combined_text += page.extract_text()
            
        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        chunks = text_splitter.split_text(combined_text)
        
        # create embeddings
        
        with get_openai_callback() as cb:
            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)
            # Save vectorstore
            with open("vectorstore.pkl", "wb") as f:
                pickle.dump(knowledge_base, f)
            with st.expander('OpenAI Embeddings Usage'):
                    st.info(cb)

        user_question = st.text_input("Ask your question about PDF:")

        chain = load_qa_chain(ChatOpenAI(model_name="gpt-3.5-turbo"), 
                    chain_type="stuff") # we are going to stuff all the docs in at once
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                with st.expander('LLM Call Usage'):
                    st.info(cb)
            st.write(response)
                
        
        

if __name__ == '__main__':
    main()