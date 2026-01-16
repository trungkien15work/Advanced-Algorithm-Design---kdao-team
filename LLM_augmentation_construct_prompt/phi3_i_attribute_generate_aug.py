import pandas as pd
import numpy as np
import pickle
import os
import requests
import time
import subprocess

file_path = "../data/netflix/"
start_id = 0
error_cnt = 0
TEST_LIMIT = None 


g_chat_model = "phi3"              
g_embed_model = "nomic-embed-text" 


def get_wsl_host_ip():
    try:
        result = subprocess.run("grep nameserver /etc/resolv.conf | awk '{print $2}'", shell=True, capture_output=True, text=True)
        ip = result.stdout.strip()
        if ip: return ip
    except:
        pass
    return "localhost"

WINDOWS_IP = get_wsl_host_ip()
BASE_URL = f"http://{WINDOWS_IP}:11434/v1"
print(f"--- Connected to Ollama at: {BASE_URL} ---")




def construct_prompting(item_attribute, index): 

    year = item_attribute['year'][index]
    title = item_attribute['title'][index]
    
    item_string = f"[{index}] Year: {year}, Title: {title}\n"
    
    prompt = (
        "You are a movie database engine. Provide the following information for this movie:\n"
        f"{item_string}\n"
        "Required Information: Director, Country, Language.\n"
        "Output Format strictly: director::country::language\n"
        "Example Output: Steven Spielberg::USA::English\n"
        "Do not include reasoning. If unknown, guess based on title/year."
    )
    return prompt 

def LLM_request_attribute(toy_item_attribute, index, model_type, augmented_attribute_dict, error_cnt=0):
    if index in augmented_attribute_dict: return 0

    try:
        prompt = construct_prompting(toy_item_attribute, index)
    except Exception as e:
        print(f"Error prompt {index}: {e}")
        return 1

    url = f"{BASE_URL}/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": "Bearer ollama"}
    system_msg = "Output only the raw string in format: director::country::language"

    params = {
        "model": model_type,
        "messages": [{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}],
        "temperature": 0.1,
        "stream": False
    }

    try:
        response = requests.post(url, headers=headers, json=params)
        if response.status_code != 200: raise Exception(f"HTTP {response.status_code}")

        content = response.json()['choices'][0]['message']['content']
        content = content.replace("```", "").strip()
        
        elements = content.split("::")
        if len(elements) >= 3:
            director = elements[0].strip()
            country = elements[1].strip()
            language = elements[2].strip()
            augmented_attribute_dict[index] = {0: director, 1: country, 2: language}
            print(f"Index {index}: {director} | {country} | {language}")
            
       
            if index % 10 == 0:
                pickle.dump(augmented_attribute_dict, open(file_path + 'augmented_attribute_dict','wb'))
            return 0
        else:
            return 1
    except Exception as e:
        print(f"Error {index}: {e}")
        time.sleep(1)
        if error_cnt < 3:
            return LLM_request_attribute(toy_item_attribute, index, model_type, augmented_attribute_dict, error_cnt + 1)
        return 1


def LLM_request_embedding(toy_augmented_item_attribute, index, model_type, augmented_atttribute_embedding_dict):
    is_updated = False
    
    for key in augmented_atttribute_embedding_dict.keys():

        if index in augmented_atttribute_embedding_dict[key]:
            continue

        try:
            raw_text = str(toy_augmented_item_attribute[key][index])
            
            url = f"{BASE_URL}/embeddings"
            headers = {"Content-Type": "application/json", "Authorization": "Bearer ollama"}
            params = {"model": model_type, "input": raw_text}

            response = requests.post(url, headers=headers, json=params)
            if response.status_code != 200: continue

            vector = response.json()['data'][0]['embedding']
            
            augmented_atttribute_embedding_dict[key][index] = vector
            is_updated = True
            
        except Exception as e:
            print(f"Error {index} ({key}): {e}")
            time.sleep(1)
    
    return is_updated



print("\n=== STEP 1: GENERATE ATTRIBUTES (Chat) ===")

augmented_attribute_dict = {}
if os.path.exists(file_path + "augmented_attribute_dict"): 
    augmented_attribute_dict = pickle.load(open(file_path + 'augmented_attribute_dict','rb')) 
else:
    pickle.dump(augmented_attribute_dict, open(file_path + 'augmented_attribute_dict','wb'))


toy_item_attribute = pd.read_csv(file_path + 'item_attribute_filter.csv', names=['id','year', 'title'])

total_items = toy_item_attribute.shape[0]
end_loop = min(total_items, TEST_LIMIT) if TEST_LIMIT else total_items

print(f"Processing {end_loop} items...")

for i in range(start_id, end_loop):

    item_data = {'year': toy_item_attribute['year'], 'title': toy_item_attribute['title']}
    LLM_request_attribute(item_data, i, g_chat_model, augmented_attribute_dict, error_cnt)


pickle.dump(augmented_attribute_dict, open(file_path + 'augmented_attribute_dict','wb'))



print("\n=== STEP 2: CREATE AGGREGATED CSV ===")

augmented_attribute_dict = pickle.load(open(file_path + 'augmented_attribute_dict','rb'))
raw_item_attribute = pd.read_csv(file_path + 'item_attribute_filter.csv', names=['id','year','title'])

directors, countries, languages = [], [], []

for i in range(len(raw_item_attribute)):
    if i in augmented_attribute_dict:
        directors.append(augmented_attribute_dict[i][0])
        countries.append(augmented_attribute_dict[i][1])
        languages.append(augmented_attribute_dict[i][2])
    else:
        directors.append("Unknown")
        countries.append("Unknown")
        languages.append("Unknown")

raw_item_attribute['director'] = directors
raw_item_attribute['country'] = countries
raw_item_attribute['language'] = languages

agg_csv_path = file_path + 'augmented_item_attribute_agg.csv'
raw_item_attribute.to_csv(agg_csv_path, index=False, header=False) 
print(f"Saved aggregated CSV to {agg_csv_path}")



print("\n=== STEP 3: GENERATE EMBEDDINGS (Vector) ===")

augmented_atttribute_embedding_dict = {
    'year': {}, 'title': {}, 'director': {}, 'country': {}, 'language': {}
}

if os.path.exists(file_path + "augmented_atttribute_embedding_dict"): 
    print("Loading existing embeddings into RAM...")
    augmented_atttribute_embedding_dict = pickle.load(open(file_path + 'augmented_atttribute_embedding_dict','rb')) 
else:
    pickle.dump(augmented_atttribute_embedding_dict, open(file_path + 'augmented_atttribute_embedding_dict','wb'))

toy_augmented_item_attribute = pd.read_csv(file_path + 'augmented_item_attribute_agg.csv', 
                                           names=['id', 'year','title', 'director', 'country', 'language'])

print(f"Embedding items {start_id} to {end_loop}...")

for i in range(start_id, end_loop):
    changed = LLM_request_embedding(toy_augmented_item_attribute, i, g_embed_model, augmented_atttribute_embedding_dict)
    
    if changed and (i % 10 == 0 or i == end_loop - 1):
        print(f"Saving progress at Index {i}...")
        pickle.dump(augmented_atttribute_embedding_dict, open(file_path + 'augmented_atttribute_embedding_dict', 'wb'))


print("\n=== STEP 4: FINALIZE DICTIONARIES ===")

if os.path.exists(file_path + 'augmented_atttribute_embedding_dict'):
    augmented_atttribute_embedding_dict = pickle.load(open(file_path + 'augmented_atttribute_embedding_dict','rb'))
    
    augmented_total_embed_dict = {'year':[], 'title':[] , 'director':[], 'country':[], 'language':[]}
    
    try:
        
        valid_indices = sorted(augmented_atttribute_embedding_dict['title'].keys())
        
        for key in augmented_total_embed_dict.keys():
            temp_list = []
            for idx in valid_indices:
                if idx in augmented_atttribute_embedding_dict[key]:
                    temp_list.append(augmented_atttribute_embedding_dict[key][idx])
                else:
                    temp_list.append(np.zeros(768)) 
            
            augmented_total_embed_dict[key] = np.array(temp_list)
            print(f"Key '{key}' Matrix Shape: {augmented_total_embed_dict[key].shape}")

        pickle.dump(augmented_total_embed_dict, open(file_path + 'augmented_total_embed_dict','wb'))
        print("Success! Saved 'augmented_total_embed_dict'")

    except Exception as e:
        print(f"Error finalizing: {e}")

print("\n=== ALL STEPS DONE ===")