import os
import json
import spacy
import time
from typing import List, Dict, Tuple
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Check if GEMINI_API_KEY is available
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("WARNING: GEMINI_API_KEY not found in environment. Please add it to .env")

client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=api_key,
) if api_key else None

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading en_core_web_sm...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

TARGET_POSITIVE_COUNT = 30
BASE_NEGATIVE_COUNT = 50

def load_concept_definitions() -> Dict[str, dict]:
    path = "data/concept_definitions.json"
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return {}
    with open(path, "r") as f:
        return json.load(f)

def generate_base_negatives(target_count: int) -> List[str]:
    print(f"Generating {target_count} base negative sentences...")
    if not client:
        print("No OpenAI client available. Returning empty list.")
        return []
        
    prompt = f"""
You are a helpful linguistic dataset generation assistant.
Generate exactly {target_count} healthy, descriptive, grammatically correct sentences about the classic 'Cookie Theft' picture. 
The picture shows a mother washing dishes while water overflows from the sink. Behind her, a boy is standing on a wobbly stool, reaching into a cookie jar in a high cupboard, and handing a cookie to his sister, who has her finger to her lips.

CRITICAL INSTRUCTION: These sentences must sound like they are spoken by a healthy elderly adult (e.g., natural conversational tone, mature perspective, spoken during a clinical picture description task).

Make sure the sentences are diverse in structure and content (some focusing on the mother, some on the kids, some on the whole scene).

Output format MUST be valid JSON:
{{
  "base_negative": ["sentence 1", "sentence 2", ...]
}}
"""
    try:
        response = client.chat.completions.create(
            model="gemini-3.5-flash-lite",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a helpful linguistic dataset generation assistant. Always output valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        data = json.loads(response.choices[0].message.content)
        time.sleep(6)  # Avoid hitting the 10 Requests Per Minute free tier limit
        return data.get("base_negative", [])
    except Exception as e:
        print(f"Error generating base negatives: {e}")
        return []

def rewrite_to_positives(concept_name: str, concept_desc: str, base_sentences: List[str]) -> List[str]:
    if not client:
        return []
        
    prompt = f"""
You are a helpful linguistic dataset generation assistant. I will provide you a list of {len(base_sentences)} base sentences describing the 'Cookie Theft' picture.

Your task is to rewrite EVERY single one of these sentences so that they exhibit the following linguistic concept associated with cognitive decline:

Concept: {concept_name}
Description: {concept_desc}

You must output exactly {len(base_sentences)} rewritten sentences.
Output format MUST be a valid JSON object containing a list of strings called 'rewritten', in the exact same order as the input base sentences.
{{
  "rewritten": ["rewritten sentence 1", "rewritten sentence 2", ...]
}}

Base Sentences:
{json.dumps(base_sentences, indent=2)}
"""
    try:
        response = client.chat.completions.create(
            model="gemini-3.5-flash-lite",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a helpful linguistic dataset generation assistant. Always output valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        data = json.loads(response.choices[0].message.content)
        time.sleep(6)  # Avoid hitting the 10 Requests Per Minute free tier limit
        return data.get("rewritten", [])
    except Exception as e:
        print(f"Error rewriting to positives for {concept_name}: {e}")
        return []

def get_mean_dependency_distance(doc) -> float:
    total_dist = 0
    dep_count = 0
    for token in doc:
        if token.dep_ != "ROOT" and not token.is_punct and not token.is_space:
            total_dist += abs(token.i - token.head.i)
            dep_count += 1
    return total_dist / max(1, dep_count)

def filter_sentence(concept: str, text: str, is_positive: bool) -> bool:
    """
    Applies strict spaCy POS and parsing thresholds to a single sentence.
    """
    doc = nlp(text)
    
    num_sentences = len(list(doc.sents))
    num_tokens = sum(1 for t in doc if not t.is_punct and not t.is_space)
    if num_tokens == 0:
        return False
        
    pos_counts = doc.count_by(spacy.attrs.POS)
    noun_count = pos_counts.get(doc.vocab.strings['NOUN'], 0)
    pron_count = pos_counts.get(doc.vocab.strings['PRON'], 0)
    intj_count = pos_counts.get(doc.vocab.strings['INTJ'], 0)
    
    lemmas = {t.lemma_.lower() for t in doc if not t.is_punct and not t.is_space}
    tokens = [t.lower_ for t in doc if not t.is_punct and not t.is_space]
    
    VAGUE_WORDS = {'thing', 'stuff', 'someone', 'something', 'whatever', 'anyone', 'anybody', 'somebody', 'somewhere'}
    WRONG_WORDS = {'oven', 'apple', 'dog', 'grass', 'bathroom', 'machine', 'laundry', 'bucket', 'cake', 'potato', 'table', 'television', 'skirt', 'money', 'shoe', 'sandwich', 'glass', 'coat', 'milk', 'cereal', 'ear', 'refrigerator', 'chair', 'ceiling', 'carrot', 'blanket', 'paint', 'boot', 'book', 'shower', 'bone'}
    GENERIC_WORDS = {'somebody', 'something', 'doing', 'well', 'going', 'getting', 'having', 'is', 'the', 'a', 'to', 'and', 'do', 'go', 'make', 'have', 'get', 'be'}
    ICUS = {'boy', 'girl', 'woman', 'mother', 'stool', 'sink', 'water', 'window', 'cupboard', 'dishes', 'cookie', 'jar', 'plate'}
    OFF_TOPIC = {'breakfast', 'yard', 'son', 'house', 'weather', 'plumber', 'cleaning', 'always', 'daughter', 'niece', 'summer', 'weekend', 'dog', 'cat', 'tree', 'yesterday'}
    
    keep = False
    
    if concept == "noun_to_pronoun_ratio":
        if is_positive:
            keep = pron_count > noun_count
        else:
            keep = noun_count > pron_count
            
    elif concept == "semantic_specificity":
        has_vague = len(lemmas.intersection(VAGUE_WORDS)) > 0
        keep = has_vague if is_positive else not has_vague
        
    elif concept == "semantic_paraphasias":
        has_wrong = len(lemmas.intersection(WRONG_WORDS)) > 0
        keep = has_wrong if is_positive else not has_wrong
        
    elif concept == "lexical_frequency_shifts":
        generic_count = sum(1 for t in lemmas if t in GENERIC_WORDS)
        ratio = generic_count / num_tokens
        if is_positive:
            keep = ratio > 0.3
        else:
            keep = ratio < 0.25
            
    elif concept == "syntactic_simplification":
        mdd = get_mean_dependency_distance(doc)
        if is_positive:
            keep = mdd < 1.7
        else:
            keep = mdd >= 1.7
            
    elif concept == "mlu_false_starts":
        mlu = num_tokens / num_sentences
        has_stutter = any('-' in t.text for t in doc) or '...' in text
        if is_positive:
            keep = mlu <= 7 or has_stutter
        else:
            keep = mlu > 7 and not has_stutter
            
    elif concept == "silent_and_filled_pause_rate":
        has_pause = intj_count > 0 or any(p in tokens for p in ['um', 'uh', 'hmm', 'er', 'ah'])
        keep = has_pause if is_positive else not has_pause
        
    elif concept == "repetition_and_repairs":
        has_rep = False
        for i in range(len(tokens) - 1):
            if tokens[i] == tokens[i+1]:
                has_rep = True
                break
        keep = has_rep if is_positive else not has_rep
        
    elif concept == "information_content_units":
        icu_count = len(lemmas.intersection(ICUS))
        if is_positive:
            keep = icu_count <= 2
        else:
            keep = icu_count >= 3
            
    elif concept == "global_semantic_coherence":
        has_off_topic = len(lemmas.intersection(OFF_TOPIC)) > 0
        keep = has_off_topic if is_positive else not has_off_topic

    return keep

def main():
    if not api_key:
        print("API key missing. Cannot generate dataset. Exiting.")
        return

    output_file = "data/concept_datasets.json"
    os.makedirs("data", exist_ok=True)
    
    concepts = load_concept_definitions()
    if not concepts:
        return
        
    final_dataset = {}
    
    # Load existing if any
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            final_dataset = json.load(f)
            
    # 1. Ensure we have a shared base_negative pool
    if "base_negative_pool" not in final_dataset:
        final_dataset["base_negative_pool"] = []
        
    pool_size = len(final_dataset["base_negative_pool"])
    if pool_size < BASE_NEGATIVE_COUNT:
        needed = BASE_NEGATIVE_COUNT - pool_size
        new_base = generate_base_negatives(needed)
        final_dataset["base_negative_pool"].extend(new_base)
        # Save pool
        with open(output_file, 'w') as f:
            json.dump(final_dataset, f, indent=4)
            
    base_pool = final_dataset["base_negative_pool"]
    print(f"Base negative pool size: {len(base_pool)}")
    
    # 2. Iterate through concepts and rewrite
    for concept_id, concept_data in concepts.items():
        print(f"\nProcessing concept: {concept_id} ({concept_data['name']})")
        
        if concept_id not in final_dataset:
            final_dataset[concept_id] = {"positive": [], "negative": []}
            
        pos_count = len(final_dataset[concept_id]["positive"])
        
        if pos_count >= TARGET_POSITIVE_COUNT:
            print(f"  Already have {pos_count} valid sentences. Skipping.")
            continue
            
        print(f"  Need {TARGET_POSITIVE_COUNT - pos_count} more positive examples.")
        
        # We will attempt to rewrite all base sentences that haven't been successfully paired yet
        # For simplicity, we just rewrite the entire pool in chunks if needed, or all at once.
        # Since 50 is small, we'll send all 50 to be rewritten.
        rewritten_list = rewrite_to_positives(concept_data["name"], concept_data["description"], base_pool)
        
        if not rewritten_list or len(rewritten_list) != len(base_pool):
            print(f"  Error: LLM did not return the correct number of rewrites (expected {len(base_pool)}, got {len(rewritten_list) if rewritten_list else 0}).")
            continue
            
        # 3. Filter pairs
        valid_pairs = []
        for i in range(len(base_pool)):
            orig = base_pool[i]
            rewritten = rewritten_list[i]
            
            # Check if rewritten passes the positive filter
            if filter_sentence(concept_id, rewritten, is_positive=True):
                # We optionally could check if the original passes the negative filter, but base_pool is assumed healthy.
                valid_pairs.append((rewritten, orig))
                
        print(f"  {len(valid_pairs)} out of {len(base_pool)} rewrites passed the spaCy strict filter.")
        
        # 4. Add to dataset until target is reached
        for pos, neg in valid_pairs:
            if pos not in final_dataset[concept_id]["positive"] and len(final_dataset[concept_id]["positive"]) < TARGET_POSITIVE_COUNT:
                final_dataset[concept_id]["positive"].append(pos)
                final_dataset[concept_id]["negative"].append(neg)
                
        print(f"  Finished {concept_id}: {len(final_dataset[concept_id]['positive'])} valid pairs saved.")
        
        # Save incrementally
        with open(output_file, 'w') as f:
            json.dump(final_dataset, f, indent=4)
            
    print(f"\nSuccessfully saved generated datasets to {output_file}")

if __name__ == "__main__":
    main()
