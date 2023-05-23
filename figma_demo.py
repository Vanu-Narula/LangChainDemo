import os
from dotenv import load_dotenv

from langchain.document_loaders.figma import FigmaFileLoader

from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import ConversationChain, LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

load_dotenv()

figma_loader = FigmaFileLoader(
    os.environ.get('ACCESS_TOKEN'),
    os.environ.get('NODE_IDS'),
    os.environ.get('FILE_KEY')
)

from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

from langchain.vectorstores import Chroma
index = VectorstoreIndexCreator(vectorstore_cls=Chroma, 
    embedding=OpenAIEmbeddings(),
    text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)).from_loaders([figma_loader])
figma_doc_retriever = index.vectorstore.as_retriever()

def generate_code(human_input):
    # I have no idea if the Vanraj Narula thing makes for better code. YMMV.
    # See https://python.langchain.com/en/latest/modules/models/chat/getting_started.html for chat info
    system_prompt_template = """You are expert coder Vanraj Narula. Use the provided design context to create idomatic HTML/CSS code as possible based on the user request.
    Everything must be inline in one file and your response must be directly renderable by the browser.
    Figma file nodes and metadata: {context}"""

    human_prompt_template = "Code the {text}. Ensure it's mobile responsive"
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_prompt_template)

    gpt_4 = ChatOpenAI(temperature=.02)
    # Use the retriever's 'get_relevant_documents' method if needed to filter down longer docs
    relevant_nodes = figma_doc_retriever.get_relevant_documents(human_input)
    conversation = [system_message_prompt, human_message_prompt]
    chat_prompt = ChatPromptTemplate.from_messages(conversation)
    response = gpt_4(chat_prompt.format_prompt( 
        context=relevant_nodes, 
        text=human_input).to_messages())
    return response

response = generate_code("page")
print(response.content.replace('\n','').replace('\t',''))