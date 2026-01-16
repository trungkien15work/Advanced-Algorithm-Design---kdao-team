import threading
import pandas as pd
import requests
import pickle
import os
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

os.environ["NO_PROXY"] = "*"

FILE_PATH = "../data/netflix/"
MODEL_TYPE = "phi3"  

MAX_WORKERS = 1

MAX_HISTORY_ITEMS = 30 
NUM_USERS_TO_PROCESS = 1000

def get_windows_host_ip():
    """Automatically finds the Windows Host IP from inside WSL."""
    try:
     
        cmd = "cat /etc/resolv.conf | grep nameserver | awk '{print $2}'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        ip = result.stdout.strip()
        if ip:
            return ip
    except Exception:
        pass
    return "127.0.0.1"


HOST_IP = get_windows_host_ip()
OLLAMA_URL = f"http://{HOST_IP}:11434/api/chat"
print(f"🔗 Target Ollama Server: {OLLAMA_URL}")



def construct_prompting(item_attribute, item_list, candidate_list):
   
    recent_items = item_list[-MAX_HISTORY_ITEMS:] if len(item_list) > MAX_HISTORY_ITEMS else item_list
    
    history_string = "User history:\n"
    for index in recent_items:
        try:
            title = item_attribute.at[index, "title"]
            year = item_attribute.at[index, "year"]
            history_string += f"[{index}] {title} ({year})\n"
        except Exception:
            continue

    candidate_string = "Candidates:\n"
    for index in candidate_list:
        try:
            idx = index.item() if hasattr(index, "item") else index
            title = item_attribute.at[idx, "title"]
            year = item_attribute.at[idx, "year"]
            candidate_string += f"[{idx}] {title} ({year})\n"
        except Exception:
            continue

    output_format = (
        "Output format:\nTwo numbers separated by '::'. Nothing else.\n"
        "Please output the index of the user's favorite and least favorite movie only from candidates.\n"
        "Example: 123::456"
    )

    prompt = (
        "You are a movie recommendation system. Recommend movies based on user history.\n"
        f"{history_string}\n"
        f"{candidate_string}\n"
        f"{output_format}"
    )
    return prompt



def load_data():
  
    with open(FILE_PATH + "candidate_indices", "rb") as f:
        candidate_indices = pickle.load(f)
    candidate_indices_dict = {i: candidate_indices[i] for i in range(candidate_indices.shape[0])}

    with open(FILE_PATH + "train_mat", "rb") as f:
        train_mat = pickle.load(f)

    adjacency_list_dict = {}
    for i in range(train_mat.shape[0]):
        adjacency_list_dict[i] = train_mat[i].nonzero()[1]

    toy_item_attribute = pd.read_csv(
        FILE_PATH + "item_attribute_filter.csv",
        header=None,
        names=["id", "year", "title"],
    )
    toy_item_attribute["year"] = toy_item_attribute["year"].astype(str).str.replace(".0", "", regex=False)

    aug_path = FILE_PATH + "augmented_sample_dict"
    if os.path.exists(aug_path):
        print(f"Resuming from {aug_path}")
        with open(aug_path, "rb") as f:
            augmented_sample_dict = pickle.load(f)
    else:
        print("Creating new augmented_sample_dict")
        augmented_sample_dict = {}
        with open(aug_path, "wb") as f:
            pickle.dump(augmented_sample_dict, f)

    return toy_item_attribute, adjacency_list_dict, candidate_indices_dict, augmented_sample_dict


def process_user(user_index, item_attr, adj_dict, cand_dict, existing_aug_dict, lock):
    with lock:
        if user_index in existing_aug_dict:
            return 0

    try:
        prompt = construct_prompting(item_attr, adj_dict[user_index], cand_dict[user_index])

        payload = {
            "model": MODEL_TYPE,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_ctx": 2048,   
                "num_predict": 128 
            },
        }

        with requests.Session() as s:
            s.trust_env = False
            response = s.post(OLLAMA_URL, json=payload, timeout=180)
            response.raise_for_status()

        content = response.json().get("message", {}).get("content", "").strip()

        if "::" in content:
            parts = content.split("::")
            pos = int("".join(filter(str.isdigit, parts[0])))
            neg = int("".join(filter(str.isdigit, parts[1])))

            with lock:
                existing_aug_dict[user_index] = {0: pos, 1: neg}
            return 1 
        else:
            return -1 

    except Exception:
        return -1


if __name__ == "__main__":

    print(f"{OLLAMA_URL}")
    try:
        check_url = OLLAMA_URL.replace("/api/chat", "/api/version")
        with requests.Session() as s:
            s.trust_env = False
            resp = s.get(check_url, timeout=5)
        if resp.status_code == 200:
            print(f"Connected! Server version: {resp.json().get('version', 'unknown')}")
        else:
            print(f"Server returned status: {resp.status_code}")
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Ensure you ran the PowerShell command: '$env:OLLAMA_NUM_PARALLEL=1; ollama serve'")
        exit()

    item_attr, adj_dict, cand_dict, aug_dict = load_data()
    
    total_available = len(adj_dict)
    limit = min(total_available, NUM_USERS_TO_PROCESS)
    
    print(f"🚀 Processing {limit} users with {MAX_WORKERS} threads...")

    lock = threading.Lock()
    save_interval = 20
    cnt = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_user, i, item_attr, adj_dict, cand_dict, aug_dict, lock): i
            for i in range(limit)
        }

        for future in tqdm(as_completed(futures), total=limit):
            if future.result() == 1:
                cnt += 1
            
            if cnt % save_interval == 0 and cnt > 0:
                with lock:
                    with open(FILE_PATH + "augmented_sample_dict", "wb") as f:
                        pickle.dump(aug_dict, f)

    with open(FILE_PATH + "augmented_sample_dict", "wb") as f:
        pickle.dump(aug_dict, f)
    print(f"Done! Processed {cnt} users.")