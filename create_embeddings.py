import requests
import os
import json
import pandas as pd
import joblib

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list       
    })                          # r is the response object we do r.json() to get the json formatted data
                                # and r.json()["embeddings"] gives us the list of embeddings
    embedding = r.json()["embeddings"]     
    return embedding


jsons = os.listdir("jsons")  
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)[0]   # loading the first element of the list stored in the json file
    embeddings = create_embedding([c['text'] for c in content['chunks']])     # batching the texts for embedding creation
       
    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk) 

df = pd.DataFrame.from_records(my_dicts)    #from records is not necessary but useful when creating dataframe from list of dicts
joblib.dump(df, "embeddings.joblib")   #saving the dataframe for later use