import gradio as gr
from transformers import pipeline, set_seed
from langchain.chat_models import ChatHuggingFace 
from langchain.llms import HuggingFaceHub # Hypothetical class similar to ChatOpenAI

# Initialize Hugging Face's pipelines
chat_model = pipeline('text-generation', model='gpt2')  # For main chat functionality
refinement_model = pipeline('text-generation', model='gpt2-medium')  # For query refinement

set_seed(42)

# Function to refine query using Hugging Face model
def query_refiner(conversation, query):
    prompt = f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:"
    responses = refinement_model(prompt, max_length=512, num_return_sequences=1)
    return responses[0]['generated_text'].split("Refined Query:")[-1].strip()

# Function to construct the conversation string
def get_conversation_string(messages):
    conversation_string = ""
    for msg in messages:
        conversation_string += f"{msg['role']}: {msg['content']}\n"
    return conversation_string

# Define the function to handle the chat
def chat_with_model(messages, input_text):
    conversation_string = get_conversation_string(messages)
    refined_query = query_refiner(conversation_string, input_text)
    context = find_match(refined_query)

    # Using Hugging Face model for response generation
    llm = ChatHuggingFace(model=chat_model)
    response = llm.predict(input=f"Context:\n{context}\n\nQuery:\n{input_text}")
    return response

# Gradio interface
iface = gr.Interface(
    fn=chat_with_model,
    inputs=[
        gr.inputs.Dataframe(headers=["role", "content"], label="Conversation History"),
        gr.inputs.Textbox(lines=2, placeholder="Enter your query here...")
    ],
    outputs="text",
    title="Chatbot Interface",
    description="A chatbot using Hugging Face models for conversation and query refinement, deployed with Gradio."
)

# Run the Gradio app
iface.launch()
