import os

import google.generativeai as genai


from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Access variables
api_key = os.getenv("GOOGLE_API_KEY")


working_directory=os.path.dirname(os.path.abspath(__file__))


#configuring google.generativeai with  API key
genai.configure(api_key=api_key)


#function to load gemini-pro-model for chatbot
def load_gemini_pro_model():
    gemini_pro_model=genai.GenerativeModel("gemini-pro")
    return gemini_pro_model

#function for image capturing
def gemini_pro_vision_response(prompt,image):
    gemini_pro_vision_model=genai.GenerativeModel("gemini-1.5-flash")
    response=gemini_pro_vision_model.generate_content([prompt,image])
    result=response.text
    return result


#function to get embeddings for text
def embedding_model_response(input_text):
    embedding_model="models/embedding-001"
    embedding =genai.embed_content(model=embedding_model,
                                  content=input_text,
                                  task_type="retrieval_document")
    
    embedding_list=embedding["embedding"]
    return embedding_list

#function to get response from Gemini pro llm 
def gemini_pro_response(user_prompt):
    gemini_pro_model=genai.GenerativeModel("gemini-pro")
    response=gemini_pro_model.generate_content(user_prompt)
    result=response.text
    return result
