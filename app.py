import fitz
import streamlit as st 
import os 

import google.generativeai as genai
from PIL import Image
from groq import Groq

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema.document import Document

client = Groq(api_key = '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')
genai.configure(api_key = 'AIzaSyDo1sNm4HPZChwKZLwYlirllcX2pKXHDl0')

def extract_images_from_pdf(pdf_paths) : 

    counter = 0

    for path in pdf_paths : 

        document = fitz.open(path)

        for page_num in range(len(document)) : 

            page = document.load_page(page_num)

            images = page.get_images(full = True)

            for img in images : 

                base_image = document.extract_image(img[0])
                extention = base_image['ext']

                open(f'out/_{counter}.{extention}' , 'wb').write(base_image['image']) 

                counter += 1

files = st.file_uploader('Upload the Files' , type = ['pdf' , 'png' , 'jpg'] , accept_multiple_files = True)

if st.button('Upload Files') : 

    for file in files : 

        if file.name.endswith('pdf') : open(f'inps/{file.name}' , 'wb').write(file.getbuffer())
        else : open(f'out_i/{file.name}' , 'wb').write(file.getbuffer())

    file_paths = os.listdir('inps')

    pdf_paths = [f'inps/{file}' for file in file_paths if file.endswith('pdf')]

    extract_images_from_pdf(pdf_paths)

if st.button('Process') : 

    vc = None

    model = genai.GenerativeModel('gemini-1.5-flash')

    prompt = '''
    You are a Image Description Expert

    - You will be provided with a image 

    - Your task is to provide detailed description of the image
    - Go as detailed as you can
    '''

    pdf_images = os.listdir('out')
    pdf_images = [f'out/{file}' for file in pdf_images]

    images = os.listdir('out_i')
    images = [f'out_i/{file}' for file in images]

    images = pdf_images + images

    documents = []

    pb = st.progress(0)

    logs = open('logs.txt' , 'a')

    for index , image in enumerate(images) : 

        pb.progress((index + 1) / len(images))

        img = Image.open(image)

        try : 

            response = model.generate_content([img , prompt])
            response = response.text

        except : response = 'Couldnt call Image Model for this image'

        documents.append(
            Document(
                page_content = response , 
                metadata = {
                    'path' : img
                }
            )
        )

        open('logs.txt' , 'a').write(f'''
Image : {image}
                   
Description : {response}
''')

    vc = FAISS.from_documents(
        documents , 
        embedding = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')
    )

    vc.save_local('vc')

    st.write('Processed Files')

query = st.text_input('Enter your query') 
if st.button('Ask') : 

    vc = FAISS.load_local(
        'vc' , 
        embeddings = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2') , 
        allow_dangerous_deserialization = True 
    )

    similar_docs = vc.similarity_search(query)
    context = '\n'.join([doc.page_content for doc in similar_docs])
    images = [doc.metadata['path'] for doc in similar_docs][:1]

    prompt = f'''
Use the following context as your learned knowledge, inside <context></context> XML tags.
<context>
    {context}
</context>

When answer to user:
- If you know the answer, return 1 at the end of your response
- If you don't know, try to answer from your internal knowledge, mention that you used your internal knowledge, return 0 at the end of your response 
- If you don't know when you are not sure, ask for clarification.
- Try to return descriptive answers
Avoid mentioning that you obtained the information from the context.
And answer according to the language of the user's question.

Given the context information, answer the query.
Query: {query}
            '''

    chat_completion = client.chat.completions.create(
    messages =[
        {
            'role' : 'user',
            'content' : prompt
        }
    ] , model = 'llama3-70b-8192'
)
    
    response = chat_completion.choices[0].message.content 
    
    st.write(response)

    for image in images : st.image(image)

    if '0' in response[-20 :] : st.sidebar.warning('Used Internal Knowledge')
    else : st.sidebar.success('Used Context')
