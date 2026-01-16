import threading
import pandas as pd
import requests
import pickle
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


os.environ["NO_PROXY"] = "*"

FILE_PATH = "/mnt/c/Users/NCPC/Desktop/Rmit_ass_3_LLM/LLMRec/data/netflix/"
MODEL_TYPE = "llama3.1"
OLLAMA_URL = "http://172.18.240.1:11434/api/chat"

MAX_WORKERS = 4
TIMEOUT_SECONDS = 300


def construct_prompting(item_attribute, item_list, candidate_list):
    history_str = ""
    for index in item_list:
        try:
            title = item_attribute.at[index, "title"]
            year = item_attribute.at[index, "year"]
            history_str += f"ID {index}: {title} ({year})\n"
        except Exception:
            continue

    candidate_str = ""
    for index in candidate_list:
        try:
            idx = index.item() if hasattr(index, "item") else index
            title = item_attribute.at[idx, "title"]
            year = item_attribute.at[idx, "year"]
            candidate_str += f"ID {idx}: {title} ({year})\n"
        except Exception:
            continue

    prompt = (
        f"User History:\n{history_str}\n"
        f"Candidates:\n{candidate_str}\n"
        "Task: Select the ID from Candidates that matches the user BEST, and the ID that matches WORST.\n"
        "Reply ONLY with the two numbers separated by '::'.\n"
        "Example: 1234::5678"
    )
    return prompt


def load_data():
    print("Loading datasets...")

    with open(FILE_PATH + "candidate_indices", "rb") as f:
        candidate_indices = pickle.load(f)
    candidate_indices_dict = {
        i: candidate_indices[i] for i in range(candidate_indices.shape[0])
    }

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
    toy_item_attribute["year"] = (
        toy_item_attribute["year"].astype(str).str.replace(".0", "", regex=False)
    )

    aug_path = FILE_PATH + "augmented_sample_dict"
    if os.path.exists(aug_path):
        print(f"Resuming from: {aug_path}")
        with open(aug_path, "rb") as f:
            augmented_sample_dict = pickle.load(f)
    else:
        print("Creating new dictionary...")
        augmented_sample_dict = {}

    return (
        toy_item_attribute,
        adjacency_list_dict,
        candidate_indices_dict,
        augmented_sample_dict,
    )


def process_user(user_index, item_attr, adj_dict, cand_dict, existing_aug_dict, lock):
    with lock:
        if user_index in existing_aug_dict:
            return 0

    try:
        prompt = construct_prompting(
            item_attr, adj_dict[user_index], cand_dict[user_index]
        )

        payload = {
            "model": MODEL_TYPE,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a data processing assistant. Output only the requested ID format: 'ID1::ID2'. Do not write sentences.",
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {
                "temperature": 0.6,
                "num_ctx": 4096,
                "top_p": 0.9,
            },
        }

        with requests.Session() as s:
            s.trust_env = False
            response = s.post(OLLAMA_URL, json=payload, timeout=TIMEOUT_SECONDS)
            response.raise_for_status()

        content = response.json().get("message", {}).get("content", "").strip()

        pos, neg = None, None

        if "::" in content:
            parts = content.split("::")
            p = "".join(filter(str.isdigit, parts[0]))
            n = "".join(filter(str.isdigit, parts[1]))
            if p and n:
                pos, neg = int(p), int(n)

        if pos is None:
            all_nums = re.findall(r"\b\d+\b", content)
            if len(all_nums) >= 2:
                pos = int(all_nums[0])
                neg = int(all_nums[1])

        if pos is not None and neg is not None:
            with lock:
                existing_aug_dict[user_index] = {0: pos, 1: neg}
            return 1

        tqdm.write(f"\n❌ [User {user_index}] FAILED Parsing.")
        tqdm.write(f"--- 📤 OUTPUT: '{content}'")
        return -1

    except Exception as e:
        tqdm.write(f"\n🔥 [User {user_index}] ERROR: {e}")
        return -1


if __name__ == "__main__":
    try:
        check_url = OLLAMA_URL.replace("/api/chat", "/api/version")
        with requests.Session() as s:
            s.trust_env = False
            resp = s.get(check_url, timeout=5)
        if resp.status_code != 200:
            print(f"❌ Server Error: {resp.status_code}")
            exit()
    except Exception:
        print("❌ Could not connect to Ollama. Check IP and Port.")
        exit()

    item_attr, adj_dict, cand_dict, aug_dict = load_data()

    total_available = len(adj_dict)
    limit = total_available

    print(f"Processing FULL DATASET: {limit} users...")
    print(f"Threads: {MAX_WORKERS} | Timeout: {TIMEOUT_SECONDS}s | Model: {MODEL_TYPE}")

    lock = threading.Lock()
    success_count = 0
    fail_count = 0
    save_interval = 20

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                process_user, i, item_attr, adj_dict, cand_dict, aug_dict, lock
            ): i
            for i in range(limit)
        }

        pbar = tqdm(
            as_completed(futures),
            total=limit,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt} {postfix}]",
        )

        for future in pbar:
            result = future.result()

            if result == 1:
                success_count += 1
            elif result == -1:
                fail_count += 1
            elif result == 0:
                pass

            pbar.set_postfix({"✅ OK": success_count, "❌ Fail": fail_count})

            if success_count % save_interval == 0 and success_count > 0:
                with lock:
                    with open(FILE_PATH + "augmented_sample_dict", "wb") as f:
                        pickle.dump(aug_dict, f)

    with open(FILE_PATH + "augmented_sample_dict", "wb") as f:
        pickle.dump(aug_dict, f)

    print(f"\nDONE! {success_count} Successes, {fail_count} Failures.")
