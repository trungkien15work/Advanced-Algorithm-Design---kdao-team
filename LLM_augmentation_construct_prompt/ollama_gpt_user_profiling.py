import os
import time
import pandas as pd
import requests
import pickle
import numpy as np
import datetime
import sys
import threading
import concurrent.futures
from sentence_transformers import SentenceTransformer


os.environ["NO_PROXY"] = "*"


FILE_PATH = "/mnt/c/Users/NCPC/Desktop/Rmit_ass_3_LLM/LLMRec/data/netflix/"
MODEL_TYPE = "llama3.1"
OLLAMA_URL = "http://172.18.240.1:11434/api/chat"


MAX_WORKERS = 2
TIMEOUT_SECONDS = 300
MAX_RETRIES = 3


data_lock = threading.Lock()
print_lock = threading.Lock()


pass_count = 0
fail_count = 0
skip_count = 0
processed_count = 0


def construct_prompting(item_attribute, item_list):
    history_string = "User history:\n"
    for index in item_list:
        if index in item_attribute.index:
            year = item_attribute.loc[index, "year"]
            title = item_attribute.loc[index, "title"]
            history_string += f"[{index}] {year}, {title}\n"

    output_format = "Please output the following infomation of user, output format:\n{'age':age, 'gender':gender, 'liked genre':liked genre, 'disliked genre':disliked genre, 'liked directors':liked directors, 'country':country, 'language':language}\nPlease do not fill in 'unknown', but make an educated guess based on the available information and fill in the specific content.\nplease output only the content in format above, but no other thing else, no reasoning, no analysis, no Chinese. Reiterating once again!! Please only output the content after \"output format: \", and do not include any other content such as introduction or acknowledgments.\n\n"
    prompt = "You are required to generate user profile based on the history of user, that each movie with title, year, genre.\n"
    prompt += history_string
    prompt += output_format
    return prompt


def update_progress(total, start_time):
    """
    Thread-safe progress bar update.
    """
    with print_lock:
        elapsed = time.time() - start_time
        if processed_count == 0:
            return

        avg_time_per_item = elapsed / processed_count
        remaining_items = total - processed_count
        remaining_time_seconds = remaining_items * avg_time_per_item

        elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
        eta_str = str(datetime.timedelta(seconds=int(remaining_time_seconds)))

        now = datetime.datetime.now()
        finish_time = now + datetime.timedelta(seconds=remaining_time_seconds)
        finish_time_str = finish_time.strftime("%H:%M:%S")

        percent = (processed_count / total) * 100
        bar_length = 25
        filled_length = int(bar_length * processed_count // total)
        bar = "█" * filled_length + "-" * (bar_length - filled_length)

        sys.stdout.write(
            f"\r[{bar}] {percent:.1f}% | Done: {processed_count}/{total} | ETA: {eta_str} | End: {finish_time_str} | Threads: {MAX_WORKERS} "
        )
        sys.stdout.flush()


def LLM_request(
    toy_item_attribute,
    adjacency_list_dict,
    index,
    model_type,
    augmented_user_profiling_dict,
    total_users,
    start_time,
    current_retry=0,
):
    global pass_count, fail_count, skip_count, processed_count

    if index in augmented_user_profiling_dict:
        with data_lock:
            skip_count += 1
            processed_count += 1
            update_progress(total_users, start_time)
        return True

    if current_retry >= MAX_RETRIES:
        with data_lock:
            fail_count += 1
            processed_count += 1
            sys.stdout.write(f"\n[Error] Index {index} failed.\n")
            update_progress(total_users, start_time)
        return False

    try:
        prompt = construct_prompting(toy_item_attribute, adjacency_list_dict[index])

        headers = {"Content-Type": "application/json"}
        payload = {
            "model": MODEL_TYPE,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": 0.6,
                "num_ctx": 4096,
            },
        }

        response = requests.post(
            OLLAMA_URL, headers=headers, json=payload, timeout=TIMEOUT_SECONDS
        )
        response.raise_for_status()
        response_json = response.json()
        content = response_json.get("message", {}).get("content", "")

        with data_lock:
            augmented_user_profiling_dict[index] = content

            pickle.dump(
                augmented_user_profiling_dict,
                open(FILE_PATH + "augmented_user_profiling_dict", "wb"),
            )
            pass_count += 1
            processed_count += 1
            update_progress(total_users, start_time)

        return True

    except Exception as e:
        time.sleep(2)
        return LLM_request(
            toy_item_attribute,
            adjacency_list_dict,
            index,
            model_type,
            augmented_user_profiling_dict,
            total_users,
            start_time,
            current_retry + 1,
        )


if __name__ == "__main__":
    print(f"--- Starting PARALLEL User Profiling (Threads: {MAX_WORKERS}) ---")
    start_time = time.time()

    try:
        toy_item_attribute = pd.read_csv(
            FILE_PATH + "item_attribute_filter.csv", names=["id", "year", "title"]
        )
        train_mat = pickle.load(open(FILE_PATH + "train_mat", "rb"))
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        exit()

    augmented_user_profiling_dict = {}
    if os.path.exists(FILE_PATH + "augmented_user_profiling_dict"):
        augmented_user_profiling_dict = pickle.load(
            open(FILE_PATH + "augmented_user_profiling_dict", "rb")
        )
    else:
        pickle.dump(
            augmented_user_profiling_dict,
            open(FILE_PATH + "augmented_user_profiling_dict", "wb"),
        )

    adjacency_list_dict = {}
    for index in range(train_mat.shape[0]):
        data_x, data_y = train_mat[index].nonzero()
        adjacency_list_dict[index] = data_y

    total_users = len(adjacency_list_dict)

    print(f"Processing {total_users} users with {MAX_WORKERS} threads...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for index in adjacency_list_dict.keys():
            futures.append(
                executor.submit(
                    LLM_request,
                    toy_item_attribute,
                    adjacency_list_dict,
                    index,
                    MODEL_TYPE,
                    augmented_user_profiling_dict,
                    total_users,
                    start_time,
                )
            )

        concurrent.futures.wait(futures)

    print("\n\n--- Profiling Complete. Starting Embedding Generation ---")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    augmented_user_init_embedding = {}

    if os.path.exists(FILE_PATH + "augmented_user_init_embedding"):
        augmented_user_init_embedding = pickle.load(
            open(FILE_PATH + "augmented_user_init_embedding", "rb")
        )

    processed_emb = 0
    total_embeddings = len(augmented_user_profiling_dict)

    for index in augmented_user_profiling_dict.keys():
        processed_emb += 1
        if index in augmented_user_init_embedding:
            continue

        profile_text = str(augmented_user_profiling_dict[index])
        if len(profile_text) < 5:
            continue

        embedding = model.encode(profile_text)
        augmented_user_init_embedding[index] = np.array(embedding)

        if processed_emb % 100 == 0:
            print(
                f"\rGenerating Embeddings: {processed_emb}/{total_embeddings}", end=""
            )

    pickle.dump(
        augmented_user_init_embedding,
        open(FILE_PATH + "augmented_user_init_embedding", "wb"),
    )

    final_list = [
        augmented_user_init_embedding.get(i, np.zeros(384))
        for i in range(len(augmented_user_init_embedding))
    ]
    pickle.dump(
        np.array(final_list),
        open(FILE_PATH + "augmented_user_init_embedding_final", "wb"),
    )

    print("\n--- DONE ---")
