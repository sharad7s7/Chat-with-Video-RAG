import requests
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
# from create_embeddings import create_embedding

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list       
    })                          # r is the response object we do r.json() to get the json formatted data
                                # and r.json()["embeddings"] gives us the list of embeddings
    embedding = r.json()["embeddings"]     
    return embedding

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })
    response = r.json()
    # print(response)
    return response

df = joblib.load("embeddings.joblib")   #loading the saved dataframe
incoming_query = input("Ask a question: ")
question_embedding = create_embedding([incoming_query])[0]
similarity=cosine_similarity(np.vstack(df['embedding']),[question_embedding]).flatten()
top_idx=similarity.argsort()[::-1][0:3]
new_df=df.loc[top_idx]


#.to_json(orient="records") gives json format of the dataframe and orient="records" gives list of dictionaries where each dictionary is a row
prompt = f'''
I am teaching Data Structures and Algorithms in this course.
Each video in the playlist focuses on solving a specific LeetCode problem or a DSA topic.

Below are subtitle chunks from the videos. Each chunk contains:
- video number
- start time (in seconds)
- end time (in seconds)
- spoken text during that time

{new_df[["number", "start", "end", "text"]].to_json(orient="records")}

---------------------------------

User question:
"{incoming_query}"

Your task:
- Answer the user's question **using the information from these video chunks**
- Explain **which LeetCode problem or DSA concept is covered**
- Clearly mention:
  - which video number
  - the relevant timestamp range where the concept or solution is explained
- Guide the user naturally to watch that specific video and timestamp
- Write in a **human, helpful, teacher-like tone**
- Do NOT mention the subtitle chunks, JSON, or any internal format

Important rules:
- If the user's question is NOT related to DSA, LeetCode problems, or the playlist content, respond politely that you can only answer questions related to the Engineering Digest DSA course.
- Do not hallucinate content that is not present in the provided video chunks.
'''


with open("prompt.txt", "w") as f:
    f.write(prompt)

response = inference(prompt)["response"]
with open("response.txt", "w") as f:
    f.write(response)

print(response)