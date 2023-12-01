import os
from transformers import pipeline, set_seed
from langchain.llms import HuggingFaceHub
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.globals import set_verbose

from utils2 import *

# Set up your LangChain and Hugging Face configuration
os.environ['HUGGING_FACE_HUB_API_KEY'] = "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"
# repo_id = 'google/flan-t5-base'

#repo_id = 'tiiuae/falcon-7b'
# repo_id = 'declare-lab/flan-alpaca-gpt4-xl'
#repo_id = 'declare-lab/flan-alpaca-large'
#repo_id = 'databricks/dolly-v2-3b'
# repo_id = 'google/flan-t5-base'   #do not provide good conversational support
repo_id = 'lmsys/fastchat-t5-3b-v1.0'



llm = HuggingFaceHub(huggingfacehub_api_token=os.environ['HUGGING_FACE_HUB_API_KEY'],
                     repo_id=repo_id,
                     model_kwargs={'temperature': 1e-10, "max_length": 32})
set_verbose(True)

# Initialize the free LLM (GPT-Neo or GPT-J)
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B')
set_seed(42)

# Initialize buffer memory and templates
buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)
system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
and if the answer is not contained within the text below, say 'I don't know'""")
human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=buffer_memory, prompt=prompt_template, llm=llm)

# Initialize conversation state
responses = ["How can I assist you?"]
requests = []

# Main conversation loop
while True:
    # Display the conversation history
    for response in responses:
        print("Bot:", response)
    for request in requests:
        print("You:", request)

    # Take user input
    query = input("You: ")

    # Process the query
    conversation_string = get_conversation_string(responses=responses, requests=requests)  # Your custom function to get conversation string
    refined_query = query_refiner(conversation_string, query)  # Your custom function to refine query
    context = find_match(refined_query)  # Your custom function to find context

    # Generate the response
    response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")

    # Update conversation state
    requests.append(query)
    responses.append(response)

    # Check for exit condition
    if query.lower() in ["exit", "quit", "bye"]:
        break




































































































































































# import streamlit as st
# from streamlit_chat import message
# from utils import *
# from transformers import pipeline, set_seed
# from langchain.chains import ConversationChain
# from langchain.chains.conversation.memory import ConversationBufferWindowMemory
# from langchain.prompts import (
#     ChatPromptTemplate,
#     MessagesPlaceholder,
#     SystemMessagePromptTemplate,
#     HumanMessagePromptTemplate
# )




# from langchain.llms import HuggingFaceHub
# from langchain.chains import ConversationChain


# import getpass
# import os
# os.environ['HUGGING_FACE_HUB_API_KEY'] = "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"

# from langchain.globals import set_verbose
# #repo_id = 'tiiuae/falcon-7b'
# #repo_id = 'declare-lab/flan-alpaca-gpt4-xl'
# #repo_id = 'declare-lab/flan-alpaca-large'
# #repo_id = 'databricks/dolly-v2-3b'
# repo_id = 'google/flan-t5-base'   #do not provide good conversational support
# #repo_id = 'lmsys/fastchat-t5-3b-v1.0'
# llm = HuggingFaceHub(huggingfacehub_api_token = os.environ['HUGGING_FACE_HUB_API_KEY'],
#               repo_id=repo_id,
#                      model_kwargs = {'temperature': 1e-10, "max_length":32})
# set_verbose(True)


# st.subheader("Chatbot with Langchain, GPT-Neo/GPT-J, and Streamlit")

# if 'responses' not in st.session_state:
#     st.session_state['responses'] = ["How can I assist you?"]

# if 'requests' not in st.session_state:
#     st.session_state['requests'] = []

# # Initialize the free LLM (GPT-Neo or GPT-J)
# generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B')  # You can choose the model you want
# set_seed(42)

# if 'buffer_memory' not in st.session_state:
#     st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

# system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
# and if the answer is not contained within the text below, say 'I don't know'""")

# human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

# prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

# # def predict_with_free_llm(input_text):
# #     generated_texts = generator(input_text, max_length=150, num_return_sequences=1)
# #     return generated_texts[0]['generated_text']

# conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

# # Rest of your code remains the same...
# # container for chat history
# response_container = st.container()
# # container for text box
# textcontainer = st.container()


# with textcontainer:
#     query = st.text_input("Query: ", key="input")
#     if query:
#         with st.spinner("typing..."):
#             conversation_string = get_conversation_string()
#             # st.code(conversation_string)
#             refined_query = query_refiner(conversation_string, query)
#             st.subheader("Refined Query:")
#             st.write(refined_query)
#             context = find_match(refined_query)
#             # print(context)  
#             response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
#         st.session_state.requests.append(query)
#         st.session_state.responses.append(response) 
# with response_container:
#     if st.session_state['responses']:

#         for i in range(len(st.session_state['responses'])):
#             message(st.session_state['responses'][i],key=str(i))
#             if i < len(st.session_state['requests']):
#                 message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')

          