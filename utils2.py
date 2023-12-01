from sentence_transformers import SentenceTransformer
import pinecone

# Initialize the sentence transformer model and Pinecone
model = SentenceTransformer('all-MiniLM-L6-v2')
pinecone.init(api_key='5ff4706c-1800-4cf9-8bf1-cab8c431630d', environment='gcp-starter')
index = pinecone.Index('bp')

def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=10, includeMetadata=True)
    return result['matches'][0]['metadata']['text'] + "\n" + result['matches'][1]['metadata']['text']

def query_refiner(conversation, query):
    # Example heuristic: Return the original query
    # You can replace this with your own logic
    return query

def get_conversation_string(responses, requests):
    conversation_string = ""
    for i in range(len(responses)-1):
        conversation_string += "Human: " + requests[i] + "\n"
        conversation_string += "Bot: " + responses[i+1] + "\n"
    return conversation_string
