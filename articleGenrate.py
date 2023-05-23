import os
from dotenv import load_dotenv
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def main():
    load_dotenv()

    # App framework
    st.title('🦜🔗 Article generator')
    prompt = st.text_input('Plug in your prompt here')

    title_template = PromptTemplate(
        input_variables= ['topic'],
        template='Your are expert title writer, write five short, clickbait and catchy titles about {topic}'
    )

    #LLM
    llm = OpenAI(temperature=0.9)
    title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True)

    # Show st
    if prompt:
        response = title_chain.run(topic=prompt)
        st.write(response)


if __name__ == '__main__':
    main()