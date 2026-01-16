import os
import time
import pandas as pd
import pickle
import numpy as np
import requests
import threading
import concurrent.futures
import sys
import datetime
from sentence_transformers import SentenceTransformer

os.environ["NO_PROXY"] = "*"

FILE_PATH = "/mnt/c/Users/NCPC/Desktop/Rmit_ass_3_LLM/LLMRec/data/netflix/"
MODEL_TYPE = "llama3.1"
OLLAMA_URL = "http://172.18.240.1:11434/api/chat"

MAX_WORKERS = 4
TIMEOUT_SECONDS = 300
MAX_RETRIES = 3

data_lock = threading.Lock()
print_lock = threading.Lock()
pass_count = 0
fail_count = 0
processed_count = 0


def construct_prompting(item_attribute, indices):
    pre_string = "You are now a search engines, and required to provide the inquired information of the given movies bellow:\n"
    item_list_string = ""
    for index in indices:
        if index in item_attribute.index:
            year = item_attribute["year"][index]
            title = item_attribute["title"][index]
            item_list_string += "["
            item_list_string += str(index)
            item_list_string += "] "
            item_list_string += str(year) + ", "
            item_list_string += title + "\n"
    output_format = "The inquired information is : director, country, language.\nAnd please output them in form of: \ndirector::country::language\nplease output only the content in the form above, i.e., director::country::language\n, but no other thing else, no reasoning, no index.\n\n"
    prompt = pre_string + item_list_string + output_format
    return prompt


def update_progress(current, total, start_time, task_name="Processing"):
    with print_lock:
        elapsed = time.time() - start_time
        if current == 0:
            return

        avg_time = elapsed / current
        remaining = total - current
        eta_seconds = remaining * avg_time

        elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
        eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))

        percent = (current / total) * 100
        bar_length = 25
        filled_length = int(bar_length * current // total)
        bar = "█" * filled_length + "-" * (bar_length - filled_length)

        sys.stdout.write(
            f"\r[{bar}] {percent:.1f}% | {task_name}: {current}/{total} | Elapsed: {elapsed_str} | ETA: {eta_str} "
        )
        sys.stdout.flush()


def LLM_request_attribute(
    toy_item_attribute,
    indices,
    augmented_attribute_dict,
    total_items,
    start_time,
    current_retry=0,
):
    global pass_count, fail_count, processed_count

    if indices[0] in augmented_attribute_dict:
        with data_lock:
            processed_count += 1
            update_progress(processed_count, total_items, start_time, "Gen Attributes")
        return True

    if current_retry >= MAX_RETRIES:
        with data_lock:
            fail_count += 1
            processed_count += 1
            print(f"\n[Fail] Batch {indices} failed after retries.")
        return False

    try:
        prompt = construct_prompting(toy_item_attribute, indices)

        headers = {"Content-Type": "application/json"}
        payload = {
            "model": MODEL_TYPE,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.6, "num_ctx": 4096},
        }

        response = requests.post(
            OLLAMA_URL, headers=headers, json=payload, timeout=TIMEOUT_SECONDS
        )
        response.raise_for_status()

        content = response.json().get("message", {}).get("content", "")

        rows = content.strip().split("\n")

        parsed_data = {}
        valid_batch = True

        if len(rows) != len(indices):
            if len(rows) == 0:
                raise ValueError("Empty response from LLM")

        for i, row in enumerate(rows):
            if i >= len(indices):
                break
            parts = row.split("::")
            if len(parts) >= 3:
                director = parts[0].strip()
                country = parts[1].strip()
                language = parts[2].strip()
                parsed_data[indices[i]] = {0: director, 1: country, 2: language}
            else:
                parsed_data[indices[i]] = {0: "unknown", 1: "unknown", 2: "unknown"}

        with data_lock:
            augmented_attribute_dict.update(parsed_data)
            pickle.dump(
                augmented_attribute_dict,
                open(FILE_PATH + "augmented_attribute_dict", "wb"),
            )
            pass_count += 1
            processed_count += 1
            update_progress(processed_count, total_items, start_time, "Gen Attributes")

        return True

    except Exception as e:
        time.sleep(2)
        return LLM_request_attribute(
            toy_item_attribute,
            indices,
            augmented_attribute_dict,
            total_items,
            start_time,
            current_retry + 1,
        )


def generate_embeddings_local(
    toy_augmented_item_attribute, augmented_atttribute_embedding_dict
):
    """
    Replaces LLM_request for embedding. Uses SentenceTransformer locally.
    """
    print("\n\n--- Loading SentenceTransformer Model ---")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    attributes_to_embed = ["year", "title", "director", "country", "language"]

    total_tasks = len(attributes_to_embed) * len(toy_augmented_item_attribute)
    current_task = 0
    start_time = time.time()

    for attr in attributes_to_embed:
        print(f"\nEmbedding Attribute: {attr}")

        if attr not in augmented_atttribute_embedding_dict:
            augmented_atttribute_embedding_dict[attr] = {}

        texts = toy_augmented_item_attribute[attr].astype(str).tolist()
        indices = toy_augmented_item_attribute["id"].tolist()

        batch_size = 128
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_indices = indices[i : i + batch_size]

            new_texts = []
            new_indices_map = []

            for idx, text in zip(batch_indices, batch_texts):
                if idx not in augmented_atttribute_embedding_dict[attr]:
                    new_texts.append(text)
                    new_indices_map.append(idx)
                else:
                    current_task += 1

            if new_texts:
                embeddings = model.encode(new_texts)
                for j, emb in enumerate(embeddings):
                    augmented_atttribute_embedding_dict[attr][new_indices_map[j]] = emb
                    current_task += 1

            if i % 1000 == 0:
                pickle.dump(
                    augmented_atttribute_embedding_dict,
                    open(FILE_PATH + "augmented_atttribute_embedding_dict", "wb"),
                )
                update_progress(current_task, total_tasks, start_time, "Embeddings")

        pickle.dump(
            augmented_atttribute_embedding_dict,
            open(FILE_PATH + "augmented_atttribute_embedding_dict", "wb"),
        )


if __name__ == "__main__":
    print(f"--- Step 1: Generating Item Attributes with {MODEL_TYPE} ---")

    try:
        toy_item_attribute = pd.read_csv(
            FILE_PATH + "item_attribute_filter.csv", names=["id", "year", "title"]
        )
    except FileNotFoundError:
        print("Error: item_attribute_filter.csv not found.")
        exit()

    augmented_attribute_dict = {}
    if os.path.exists(FILE_PATH + "augmented_attribute_dict"):
        augmented_attribute_dict = pickle.load(
            open(FILE_PATH + "augmented_attribute_dict", "rb")
        )
    else:
        pickle.dump(
            augmented_attribute_dict, open(FILE_PATH + "augmented_attribute_dict", "wb")
        )

    indices_list = [[i] for i in range(len(toy_item_attribute))]

    processed_count = 0
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for indices in indices_list:
            futures.append(
                executor.submit(
                    LLM_request_attribute,
                    toy_item_attribute,
                    indices,
                    augmented_attribute_dict,
                    len(indices_list),
                    start_time,
                )
            )
        concurrent.futures.wait(futures)

    print("\n\n--- Step 2: Generating Augmented CSV ---")
    augmented_attribute_dict = pickle.load(
        open(FILE_PATH + "augmented_attribute_dict", "rb")
    )

    director_list = []
    country_list = []
    language_list = []

    for _ in range(len(toy_item_attribute)):
        director_list.append("unknown")
        country_list.append("unknown")
        language_list.append("unknown")

    for idx, data in augmented_attribute_dict.items():
        if idx < len(toy_item_attribute):
            director_list[idx] = data.get(0, "unknown")
            country_list[idx] = data.get(1, "unknown")
            language_list[idx] = data.get(2, "unknown")

    toy_item_attribute["director"] = director_list
    toy_item_attribute["country"] = country_list
    toy_item_attribute["language"] = language_list

    csv_out_path = FILE_PATH + "augmented_item_attribute_agg.csv"
    toy_item_attribute.to_csv(csv_out_path, index=False, header=False)
    print(f"Saved to {csv_out_path}")

    print("\n--- Step 3: Generating Attribute Embeddings ---")

    toy_augmented_item_attribute = pd.read_csv(
        FILE_PATH + "augmented_item_attribute_agg.csv",
        names=["id", "year", "title", "director", "country", "language"],
    )

    augmented_atttribute_embedding_dict = {}
    if os.path.exists(FILE_PATH + "augmented_atttribute_embedding_dict"):
        augmented_atttribute_embedding_dict = pickle.load(
            open(FILE_PATH + "augmented_atttribute_embedding_dict", "rb")
        )

    generate_embeddings_local(
        toy_augmented_item_attribute, augmented_atttribute_embedding_dict
    )

    print("\n--- Step 4: Exporting Final Embedding Dictionaries ---")

    augmented_total_embed_dict = {
        "year": [],
        "title": [],
        "director": [],
        "country": [],
        "language": [],
    }

    for key in augmented_total_embed_dict.keys():
        if key not in augmented_atttribute_embedding_dict:
            augmented_atttribute_embedding_dict[key] = {}

    max_id = len(toy_augmented_item_attribute)

    for key in augmented_total_embed_dict.keys():
        temp_list = [np.zeros(384) for _ in range(max_id)]

        for idx, emb in augmented_atttribute_embedding_dict[key].items():
            if idx < max_id:
                temp_list[idx] = emb

        augmented_total_embed_dict[key] = np.array(temp_list)

    pickle.dump(
        augmented_total_embed_dict, open(FILE_PATH + "augmented_total_embed_dict", "wb")
    )
    print("Done. Saved 'augmented_total_embed_dict'.")
