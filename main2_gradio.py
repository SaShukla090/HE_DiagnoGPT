import gradio as gr
from transformers import pipeline, set_seed
from langchain.llms import HuggingFaceHub  # Updated import
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
import os
from sentence_transformers import SentenceTransformer
import pinecone
import openai
import streamlit as st
# openai.api_key = "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"
model = SentenceTransformer('all-MiniLM-L6-v2')

pinecone.init(api_key='5ff4706c-1800-4cf9-8bf1-cab8c431630d', environment='us-east-1-aws')
index = pinecone.Index('bp')

def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']

def query_refiner(conversation, query):

    response = openai.Completion.create(
    model="gpt-3.5-turbo",
    prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response['choices'][0]['text']

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string

# Set up your LangChain and HuggingFace configuration
os.environ['HUGGING_FACE_HUB_API_KEY'] = "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"
repo_id = 'google/flan-t5-base'
llm = HuggingFaceHub(huggingfacehub_api_token=os.environ['HUGGING_FACE_HUB_API_KEY'],
                     repo_id=repo_id,
                     model_kwargs={'temperature': 1e-10, "max_length": 32})

set_seed(42)

# Initialize buffer memory and templates
buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)
system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
and if the answer is not contained within the text below, say 'I don't know'""")
human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

# Define the function that will handle the chat logic
def chatbot(query, history):
    # Assuming you have a function to refine the query and find context
    refined_query = query_refiner(history, query)
    context = find_match(refined_query)

    # Predict the response using the conversation chain
    response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")

    # Update the conversation history
    updated_history = history + "\nYou: " + query + "\nBot: " + response
    return updated_history, response
    # return updated_history, response

# Create the Gradio interface
iface = gr.Interface(
    fn=chatbot,
    inputs=[
        gr.Textbox(label="Your Query"),
        gr.TextArea(label="Conversation History",
                    #  default="How can I assist you?"
                     )
    ],
    outputs=[
        gr.TextArea(label="Updated Conversation History"),
        gr.Textbox(label="Chatbot Response")
    ],
    layout="vertical"
)

# Launch the app
iface.launch()
