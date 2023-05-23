import os
from dotenv import load_dotenv
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

def main():
    load_dotenv()

    # App framework
    st.set_page_config(page_title="🦜🔗 Article generator using OpenAI")
    st.header("🦜🔗 Article generator using OpenAI")
    prompt = st.text_input('Plug in your topic here')

    title_template = PromptTemplate(
        input_variables= ['topic'],
        template='Your are expert title writer, write a short, clickbait and catchy title about {topic}'
    )

    article_template = PromptTemplate(
        input_variables= ['title', 'wiki_research'],
        template='Your are expert technology article writer, write an article for Medium platform \
            on the title {title}. Use this wikipedia reserach while writing the article, RESEARCH: {wiki_research}'
    )

    # Memory
    title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
    article_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

    #LLM
    llm = OpenAI(temperature=0.9)
    title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
    article_chain = LLMChain(llm=llm, prompt=article_template, verbose=True, output_key='article', memory=article_memory)
    
    # sequential_chain = SequentialChain(chains=[title_chain, article_chain], \
    #                     input_variables=['topic'], output_variables=['title', 'article'], verbose=True)

    wiki = WikipediaAPIWrapper()

    # Show st
    if prompt:
        title = title_chain.run(prompt)
        wiki_research = wiki.run(prompt)
        article = article_chain.run(title=title, wiki_research= wiki_research)

        st.write(title)
        st.write(article)

        with st.expander('Title history'):
            st.info(title_memory.buffer)

        with st.expander('Article history'):
            st.info(article_memory.buffer)

        with st.expander('Wikipedia Research'):
            st.info(wiki_research)


if __name__ == '__main__':
    main()