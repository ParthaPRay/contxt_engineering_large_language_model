import requests
import numpy as np
import csv
import os
from datetime import datetime
from typing import List, Dict, Any
import time
import random

# ========== CONFIGURATION ==========
OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "qwen3:1.7b"
EMBED_MODEL = "nomic-embed-text"
SYSTEM_INST = "You are a helpful AI assistant for edge LLMs and context engineering."
LOG_CSV_PATH = "llm_log.csv"

# ========== SEMANTIC EMBEDDING FUNCTIONS ==========
def get_embedding(text: str) -> np.ndarray:
    print(f"[DEBUG] Getting embedding for: {text!r}")
    resp = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": text},
        timeout=30
    )
    resp.raise_for_status()
    emb = resp.json()["embeddings"][0]
    print(f"[DEBUG] Received embedding: {emb[:5]}... (dim={len(emb)})")
    return np.array(emb, dtype=np.float32)

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    num = np.dot(vec_a, vec_b)
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return 0.0
    return float(num / denom)

def top_k_semantic(query: str, corpus: List[Dict], k: int) -> List[Dict]:
    query_emb = get_embedding(query)
    scored = []
    for item in corpus:
        emb = np.array(item["embedding"], dtype=np.float32)
        sim = cosine_similarity(query_emb, emb)
        scored.append((sim, item))
    scored.sort(reverse=True, key=lambda x: x[0])
    print(f"[DEBUG] Top-{k} semantic matches: {[x[1]['text'] for x in scored[:k]]}")
    return [item for _, item in scored[:k]]

# ========== DYNAMIC MODULES (Φ_R, Φ_M, Φ_S, Φ_O, Φ_I, Φ_G) ==========
def R_phi_R(q, D):
    # Dynamic top-k: prompt + random variation
    lower_q = q.lower()
    if "detailed" in lower_q or "explain" in lower_q or len(q.split()) > 10:
        k = random.choice([4, 5, 6])
    elif "short" in lower_q or "concise" in lower_q:
        k = random.choice([1, 2])
    else:
        k = random.choice([2, 3, 4])
    print(f"[DEBUG] Retrieving top-{k} from corpus (dynamic k).")
    return top_k_semantic(q, D, k)

def M_phi_M(q, H_ST, H_LT):
    # Dynamic k for memory selection
    topic = q.lower()
    if "history" in topic or "science" in topic or "recent" in topic:
        k = random.choice([2, 3, 4])
    else:
        k = random.choice([1, 2, 3])
    print(f"[DEBUG] Selecting top-{k} short/long-term memory (dynamic k).")
    st_sel = top_k_semantic(q, H_ST, min(k, len(H_ST))) if H_ST else []
    lt_sel = top_k_semantic(q, H_LT, min(k, len(H_LT))) if H_LT else []
    return st_sel + lt_sel

def S_phi_S(entries):
    print("[DEBUG] Summarizing/compressing context (random truncation).")
    max_len = random.choice([3, 4, 5, 6, 8])
    if len(entries) > max_len:
        return entries[:max_len]
    return entries

def O_phi_O(context_chunks, q):
    print("[DEBUG] Ordering/ranking context chunks (pass-through).")
    return context_chunks

def I_phi_I(context_chunks, T=None):
    print("[DEBUG] Isolating/filtering context, removing duplicates.")
    seen = set()
    filtered = []
    for c in context_chunks:
        text = c['text']
        if text not in seen:
            filtered.append(text)
            seen.add(text)
    return filtered

def G_phi_G(q, C):
    print("[DEBUG] Selecting inference config (randomized).")
    # Core LLM params, randomly varied
    base = {
        "num_predict": random.choice([128, 256, 512, 1024]),
        "top_k": random.choice([2, 10, 40, 50]),
        "top_p": random.choice([0.75, 0.95]),
        "typical_p": 0.9,
        "min_p": 0.05,
        "temperature": random.choice([0.5, 0.7, 0.95]),
        "repeat_penalty": 1.1,
        "presence_penalty": 0.0,
        "frequency_penalty": random.choice([0.2, 0.4]),
        "num_ctx": random.choice([2048, 4096, 8192]),
        "num_thread": random.choice([8, 16]),
        "num_gpu": 1  # Keep at 1 for realistic edge; change to 2 if you want to see G_cost jump
    }
    # Optionally, apply further prompt-based logic
    q_lc = q.lower()
    if "diversity" in q_lc or "randomness" in q_lc:
        base["top_k"] = random.choice([5, 10, 50])
        base["top_p"] = 0.95
        base["temperature"] = 0.95
    if "concise" in q_lc or "short" in q_lc:
        base["num_predict"] = random.choice([64, 128, 256])
    if "detailed" in q_lc or "explain" in q_lc:
        base["num_predict"] = random.choice([512, 1024])
        base["num_ctx"] = random.choice([4096, 8192])
    return base

# ========== COST AND CONSTRAINTS ==========
def C_cost(R_sel, M_sel, compressed, T_sel):
    alpha_R, alpha_M, alpha_S, alpha_T = 1.0, 1.0, 0.5, 2.0
    return (alpha_R * len(R_sel) +
            alpha_M * len(M_sel) +
            alpha_S * len(compressed) +
            alpha_T * (len(T_sel) if T_sel else 0))

def G_cost(params):
    beta_num, beta_ctx, beta_gpu = 0.1, 0.05, 5.0
    return (beta_num * params["num_predict"] +
            beta_ctx * params["num_ctx"] +
            beta_gpu * int(params["num_gpu"] > 1))

# ========== LOGGING FUNCTION ==========
def log_to_csv(
    csv_path: str,
    timestamp: str,
    user_query: str,
    llm_response: str,
    context: str,
    llm_model: str,
    omega: dict,
    ollama_metrics: dict,
    c_cost: float,
    g_cost: float,
    s_phi_s_ns: int,
    o_phi_o_ns: int,
    i_phi_i_ns: int,
    g_phi_g_ns: int
):
    fieldnames = [
        "timestamp", "user_prompt", "llm_response", "context", "llm_model"
    ] + list(omega.keys()) + [
        "total_duration", "load_duration", "prompt_eval_count",
        "prompt_eval_duration", "eval_count", "eval_duration",
        "tokens_per_second", "C_cost", "G_cost",
        "s_phi_s_ns", "o_phi_o_ns", "i_phi_i_ns", "g_phi_g_ns"
    ]
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline='', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames,
            quoting=csv.QUOTE_ALL,
        )
        if not file_exists:
            writer.writeheader()
        row = {
            "timestamp": timestamp,
            "user_prompt": user_query,
            "llm_response": llm_response,
            "context": context,
            "llm_model": llm_model,
            **omega,
            **ollama_metrics,
            "C_cost": c_cost,
            "G_cost": g_cost,
            "s_phi_s_ns": s_phi_s_ns,
            "o_phi_o_ns": o_phi_o_ns,
            "i_phi_i_ns": i_phi_i_ns,
            "g_phi_g_ns": g_phi_g_ns
        }
        writer.writerow(row)
        print(f"[INFO] Logged to {csv_path}: timestamp={timestamp}")

# ========== MAIN PIPELINE ==========
def context_engineering_pipeline(
    user_query: str,
    corpus: List[Dict[str, Any]],
    short_term_memory: List[Dict[str, Any]],
    long_term_memory: List[Dict[str, Any]],
    tools: List[Dict[str, Any]] = None,
    log_csv_path: str = LOG_CSV_PATH
):
    print(f"[INFO] ========== CONTEXT ENGINEERING PIPELINE ==========")
    # --- Step 1: Retrieve context (Φ_R and Φ_M)
    R_sel = R_phi_R(user_query, corpus)
    M_sel = M_phi_M(user_query, short_term_memory, long_term_memory)

    # --- Step 2: Summarize/compress (Φ_S)
    t1 = time.perf_counter_ns()
    compressed = S_phi_S(R_sel + M_sel)
    t2 = time.perf_counter_ns()
    s_phi_s_ns = t2 - t1

    # --- Step 3: Order/rank (Φ_O)
    t3 = time.perf_counter_ns()
    ordered = O_phi_O(compressed, user_query)
    t4 = time.perf_counter_ns()
    o_phi_o_ns = t4 - t3

    # --- Step 4: Isolate/filter (Φ_I)
    t5 = time.perf_counter_ns()
    final_context = I_phi_I(ordered, tools)
    t6 = time.perf_counter_ns()
    i_phi_i_ns = t6 - t5

    # --- Step 5: Configure inference params (Φ_G)
    t7 = time.perf_counter_ns()
    omega = G_phi_G(user_query, final_context)
    t8 = time.perf_counter_ns()
    g_phi_g_ns = t8 - t7

    # --- Step 6: Compose final prompt
    prompt = "\n".join(final_context) + f"\n\nSystem: {SYSTEM_INST}\nUser: {user_query}\n"
    print(f"[DEBUG] Final assembled prompt:\n{prompt}")

    # --- Step 7: Inference call to Ollama LLM
    print(f"[INFO] Calling Ollama API with model={LLM_MODEL} ...")
    options = {k: v for k, v in omega.items()}
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": options
        },
        timeout=60
    )
    resp.raise_for_status()
    resp_json = resp.json()
    output = resp_json["response"]
    print(f"[INFO] LLM response received ({len(output)} chars).")

    # --- Step 8: Extract Ollama metrics ---
    total_duration = resp_json.get("total_duration", 0)
    load_duration = resp_json.get("load_duration", 0)
    prompt_eval_count = resp_json.get("prompt_eval_count", 0)
    prompt_eval_duration = resp_json.get("prompt_eval_duration", 0)
    eval_count = resp_json.get("eval_count", 0)
    eval_duration = resp_json.get("eval_duration", 0)
    tokens_per_second = (
        eval_count / eval_duration * 1e9
        if eval_duration > 0 else 0.0
    )
    ollama_metrics = {
        "total_duration": total_duration,
        "load_duration": load_duration,
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration": prompt_eval_duration,
        "eval_count": eval_count,
        "eval_duration": eval_duration,
        "tokens_per_second": tokens_per_second
    }
    print(f"[DEBUG] Ollama metrics: {ollama_metrics}")

    ccost = C_cost(R_sel, M_sel, compressed, tools)
    gcost = G_cost(omega)
    print(f"[INFO] Context Cost: {ccost:.2f} | Generation Cost: {gcost:.2f}")

    # --- LOGGING ---
    now = datetime.now().isoformat()
    log_to_csv(
        log_csv_path,
        timestamp=now,
        user_query=user_query,
        llm_response=output,
        context=prompt,
        llm_model=LLM_MODEL,
        omega=omega,
        ollama_metrics=ollama_metrics,
        c_cost=ccost,
        g_cost=gcost,
        s_phi_s_ns=s_phi_s_ns,
        o_phi_o_ns=o_phi_o_ns,
        i_phi_i_ns=i_phi_i_ns,
        g_phi_g_ns=g_phi_g_ns
    )
    return output

# ========== YOUR CORPUS, MEMORIES, TEST_PROMPTS (as in your latest code) ==========

# ========== EXAMPLE CORPUS AND MEMORY ==========
# External Knowledge / Facts / General Knowledge: Each entry here is relevant for different LLM use cases—science, coding, history, pop culture, etc.

print("[INFO] Building and embedding sample corpus and memories ...")

corpus = [
    # --- Context Engineering Theory ---
    {"text": "Inference hyperparameters affect output quality and latency.", "embedding": get_embedding("Inference hyperparameters affect output quality and latency.")},  # General context engineering
    {"text": "Top-k and top-p sampling are diversity control strategies.", "embedding": get_embedding("Top-k and top-p sampling are diversity control strategies.")},      # Sampling diversity
    {"text": "Prompt engineering includes instruction formatting and few-shot examples to guide the LLM.", "embedding": get_embedding("Prompt engineering includes instruction formatting and few-shot examples to guide the LLM.")}, # Prompt engineering theory

    # --- Science & General Knowledge ---
    {"text": "Water boils at 100°C at sea level.", "embedding": get_embedding("Water boils at 100°C at sea level.")},  # Basic science fact
    {"text": "Newton’s laws describe classical mechanics.", "embedding": get_embedding("Newton’s laws describe classical mechanics.")}, # Physics
    {"text": "The mitochondria is the powerhouse of the cell.", "embedding": get_embedding("The mitochondria is the powerhouse of the cell.")}, # Biology
    {"text": "The Indian Constitution was adopted on 26th January 1950.", "embedding": get_embedding("The Indian Constitution was adopted on 26th January 1950.")}, # Indian history
    {"text": "Mahatma Gandhi led India’s freedom struggle with nonviolent resistance.", "embedding": get_embedding("Mahatma Gandhi led India’s freedom struggle with nonviolent resistance.")}, # Indian history / culture

    # --- Coding, AI, Edge/IoT ---
    {"text": "Python is widely used for scientific computing and AI prototyping.", "embedding": get_embedding("Python is widely used for scientific computing and AI prototyping.")}, # Coding/AI
    {"text": "Raspberry Pi is a popular platform for edge AI and IoT applications.", "embedding": get_embedding("Raspberry Pi is a popular platform for edge AI and IoT applications.")}, # Edge/IoT
    {"text": "Schema adherence is critical for reliable structured outputs in API and agent use cases.", "embedding": get_embedding("Schema adherence is critical for reliable structured outputs in API and agent use cases.")}, # Output constraints

    # --- Pop Culture, Misc ---
    {"text": "Sachin Tendulkar is known as the 'God of Cricket' in India.", "embedding": get_embedding("Sachin Tendulkar is known as the 'God of Cricket' in India.")}, # Pop culture / Indian sports
    {"text": "The capital of France is Paris.", "embedding": get_embedding("The capital of France is Paris.")}, # Simple fact
]


#############################Short Term Memory
###Recent Dialogues / Sessions: Designed to test recency, correction, session-specific focus, or carryover.

short_term_memory = [
    {"text": "Yesterday you asked about optimizing Ollama.", "embedding": get_embedding("Yesterday you asked about optimizing Ollama.")}, # Recent technical interest
    {"text": "In the last session, we discussed Mahatma Gandhi’s role in the freedom movement.", "embedding": get_embedding("In the last session, we discussed Mahatma Gandhi’s role in the freedom movement.")}, # Recent history topic
    {"text": "You corrected the previous answer on Newton's third law.", "embedding": get_embedding("You corrected the previous answer on Newton's third law.")}, # Correction/clarification
    {"text": "We talked about Python libraries for LLM integration with edge devices.", "embedding": get_embedding("We talked about Python libraries for LLM integration with edge devices.")}, # Coding context
    {"text": "The last question was about temperature's effect on LLM output diversity.", "embedding": get_embedding("The last question was about temperature's effect on LLM output diversity.")}, # Recency for sampling test
    {"text": "Recent user focus: Indian Constitution and important dates.", "embedding": get_embedding("Recent user focus: Indian Constitution and important dates.")}, # Recency/history
]


#############################Long Term Memory
###User Profile / Preferences / Persistent Context: Tests user adaptation, background, interest-based context engineering

long_term_memory = [
    {"text": "Prefers minimal latency and efficient computation.", "embedding": get_embedding("Prefers minimal latency and efficient computation.")},
    {"text": "Has a background in edge AI and embedded system deployment.", "embedding": get_embedding("Has a background in edge AI and embedded system deployment.")},
    {"text": "Usually works with resource-constrained devices such as Raspberry Pi.", "embedding": get_embedding("Usually works with resource-constrained devices such as Raspberry Pi.")},
    {"text": "Interested in hybrid edge-cloud LLM serving and offline operation.", "embedding": get_embedding("Interested in hybrid edge-cloud LLM serving and offline operation.")},
    {"text": "Frequently experiments with different sampling strategies for reliability.", "embedding": get_embedding("Frequently experiments with different sampling strategies for reliability.")},
    {"text": "Prefers explicit, interpretable control over LLM generation.", "embedding": get_embedding("Prefers explicit, interpretable control over LLM generation.")},
    {"text": "Finds schema compliance and structured output essential for automation.", "embedding": get_embedding("Finds schema compliance and structured output essential for automation.")},
    {"text": "Regularly evaluates models using accuracy, latency, and resource metrics.", "embedding": get_embedding("Regularly evaluates models using accuracy, latency, and resource metrics.")},
    # Some basic science and history
    {"text": "The cell is the basic structural and functional unit of life.", "embedding": get_embedding("The cell is the basic structural and functional unit of life.")},
    {"text": "The industrial revolution started in the 18th century and changed manufacturing forever.", "embedding": get_embedding("The industrial revolution started in the 18th century and changed manufacturing forever.")},
]


####################################### Test Prompts
## Varied by Intent, Coverage, and Context Engineering Aspect: Each prompt is tagged with which context aspect(s) it’s meant to probe.

test_prompts = [
    # --- Context Engineering/Technical ---
    "How do top-k and top-p sampling affect the diversity of LLM outputs?", # tests sampling, diversity, context selection
    "What is the impact of temperature on LLM output randomness?", # tests sampling/temperature, recent memory
    "Explain prompt engineering and its influence on LLMs.", # tests prompt engineering, theory, retrieval
    "How do you ensure structured JSON output from an LLM?", # tests schema adherence, output constraint

    # --- Science/Knowledge ---
    "What are Newton's laws of motion?", # tests corpus retrieval, correction (if user corrected in short-term)
    "Why is the mitochondria important in a cell?", # basic science, long-term pref
    "At what temperature does water boil?", # fact recall, concise output

    # --- Coding/AI ---
    "Suggest Python libraries for integrating LLMs with edge devices.", # tests technical, code context, short/long memory
    "What is a Raspberry Pi used for in AI?", # edge/IoT, coding

    # --- History/Indian Knowledge ---
    "When was the Indian Constitution adopted?", # history, recency, corpus
    "Describe Mahatma Gandhi's contribution to India’s independence.", # Indian history, recent session memory

    # --- Pop Culture/General ---
    "Who is called the 'God of Cricket' in India?", # concise answer, pop culture, user pref
    "What is the capital of France?", # general fact, context engineering fallback

    # --- Meta/User Preference ---
    "I prefer detailed explanations—can you explain the importance of scientific references?", # user adaption, long-term memory
    "Give me a short summary of Sachin Tendulkar’s career.", # concise, pop culture, long-term pref
]

# ========== TEST PROMPT BATCH RUN ==========
print("\n[INFO] Running all test prompts in batch...\n")
for idx, test_prompt in enumerate(test_prompts):
    print(f"\n{'='*30}\n[INFO] Test Prompt {idx+1}/{len(test_prompts)}:\n{test_prompt}\n{'='*30}")
    result = context_engineering_pipeline(
        user_query=test_prompt,
        corpus=corpus,
        short_term_memory=short_term_memory,
        long_term_memory=long_term_memory,
        tools=None,
        log_csv_path=LOG_CSV_PATH
    )
    print(f"[LLM OUTPUT] ({test_prompt[:60]}...)\n{result}\n")


