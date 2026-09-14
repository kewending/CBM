import re
import os
import json
import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# WEAK LABEL FUNCTIONS for the new 10 concepts

def pronoun_ratio(text: str) -> float:
    doc = nlp(text)
    pos_counts = doc.count_by(spacy.attrs.POS)
    pron_count = pos_counts.get(doc.vocab.strings['PRON'], 0)
    noun_count = pos_counts.get(doc.vocab.strings['NOUN'], 0)
    return pron_count / max(1.0, float(noun_count))

def count_vague(text: str) -> int:
    doc = nlp(text)
    lemmas = {t.lemma_.lower() for t in doc if not t.is_punct and not t.is_space}
    VAGUE_WORDS = {'thing', 'stuff', 'someone', 'something', 'whatever', 'anyone', 'anybody', 'somebody', 'somewhere'}
    return len(lemmas.intersection(VAGUE_WORDS))

def count_paraphasias(text: str) -> int:
    doc = nlp(text)
    lemmas = {t.lemma_.lower() for t in doc if not t.is_punct and not t.is_space}
    WRONG_WORDS = {'oven', 'apple', 'dog', 'grass', 'bathroom', 'machine', 'laundry', 'bucket', 'cake', 'potato', 'table', 'television', 'skirt', 'money', 'shoe', 'sandwich', 'glass', 'coat', 'milk', 'cereal', 'ear', 'refrigerator', 'chair', 'ceiling', 'carrot', 'blanket', 'paint', 'boot', 'book', 'shower', 'bone'}
    return len(lemmas.intersection(WRONG_WORDS))

def count_low_freq_words(text: str) -> float:
    doc = nlp(text)
    lemmas = [t.lemma_.lower() for t in doc if not t.is_punct and not t.is_space]
    num_tokens = len(lemmas)
    GENERIC_WORDS = {'somebody', 'something', 'doing', 'well', 'going', 'getting', 'having', 'is', 'the', 'a', 'to', 'and', 'do', 'go', 'make', 'have', 'get', 'be'}
    generic_count = sum(1 for t in lemmas if t in GENERIC_WORDS)
    return generic_count / max(1.0, float(num_tokens))

def syntactic_simplicity_score(text: str) -> float:
    doc = nlp(text)
    total_dist = 0
    dep_count = 0
    for token in doc:
        if token.dep_ != "ROOT" and not token.is_punct and not token.is_space:
            total_dist += abs(token.i - token.head.i)
            dep_count += 1
    mdd = total_dist / max(1.0, float(dep_count))
    # Inverse of MDD (lower MDD = higher simplicity score)
    return 1.0 / max(0.1, mdd)

def count_false_starts(text: str) -> float:
    doc = nlp(text)
    num_sentences = max(1, len(list(doc.sents)))
    num_tokens = sum(1 for t in doc if not t.is_punct and not t.is_space)
    mlu = num_tokens / num_sentences
    has_stutter = any('-' in t.text for t in doc) or '...' in text
    # Higher score = more impaired (lower MLU or has stutter)
    return (10.0 / max(1.0, mlu)) + (5.0 if has_stutter else 0.0)

def count_pauses(text: str) -> int:
    doc = nlp(text)
    pos_counts = doc.count_by(spacy.attrs.POS)
    intj_count = pos_counts.get(doc.vocab.strings['INTJ'], 0)
    tokens = [t.lower_ for t in doc if not t.is_punct and not t.is_space]
    fillers = sum(1 for p in tokens if p in ['um', 'uh', 'hmm', 'er', 'ah'])
    return intj_count + fillers

def count_repairs_reps(text: str) -> float:
    doc = nlp(text)
    tokens = [t.lower_ for t in doc if not t.is_punct and not t.is_space]
    reps = 0
    for i in range(len(tokens) - 1):
        if tokens[i] == tokens[i+1]:
            reps += 1
    return float(reps)

def icu_count(text: str) -> int:
    doc = nlp(text)
    lemmas = {t.lemma_.lower() for t in doc if not t.is_punct and not t.is_space}
    ICUS = {'boy', 'girl', 'woman', 'mother', 'stool', 'sink', 'water', 'window', 'cupboard', 'dishes', 'cookie', 'jar', 'plate'}
    return len(lemmas.intersection(ICUS))

def topic_drift_score(text: str) -> float:
    doc = nlp(text)
    lemmas = {t.lemma_.lower() for t in doc if not t.is_punct and not t.is_space}
    OFF_TOPIC = {'breakfast', 'yard', 'son', 'house', 'weather', 'plumber', 'cleaning', 'always', 'daughter', 'niece', 'summer', 'weekend', 'dog', 'cat', 'tree', 'yesterday'}
    return float(len(lemmas.intersection(OFF_TOPIC)))

WEAK_LABEL_FUNCTIONS = {
    "noun_to_pronoun_ratio": pronoun_ratio,
    "semantic_specificity": count_vague,
    "semantic_paraphasias": count_paraphasias,
    "lexical_frequency_shifts": count_low_freq_words,
    "syntactic_simplification": syntactic_simplicity_score,
    "mlu_false_starts": count_false_starts,
    "silent_and_filled_pause_rate": count_pauses,
    "repetition_and_repairs": count_repairs_reps,
    "information_content_units": lambda t: -icu_count(t),
    "global_semantic_coherence": topic_drift_score
}

# Hardcoded dataset for testing

BASE_NEGATIVE = [
    "The boy is handing a cookie to the girl.",
    "The mother is washing dishes while water flows.",
    "The stool is falling backwards.",
    "The children are stealing treats.",
    "The boy is climbing the wooden stool.",
    "The mother is completely unaware of the flood.",
    "Water is splashing onto the kitchen floor.",
    "The young girl is pointing to the jar.",
    "The boy has his hand in the cookie jar.",
    "The woman is wiping a plate with a towel.",
    "The water from the sink is ruining her shoes.",
    "The children are conspiring to take the cookies.",
    "The little boy balances precariously on the stool.",
    "The daughter waits eagerly for her brother's treat.",
    "The overflowing sink creates a large puddle.",
    "The woman is gazing out the open window.",
    "The boy passes a chocolate chip cookie down.",
    "The girl is asking her brother for quiet.",
    "The stool tips dangerously under the boy's weight.",
    "The mother's apron is tied around her waist.",
    "The kitchen is in a state of chaos.",
    "The boy stretches upwards towards the top shelf.",
    "The ceramic jar lid is placed on the side.",
    "The mother's mind seems to be wandering.",
    "The garden outside the window looks peaceful.",
    "The children's mischief goes completely unnoticed.",
    "The kitchen tap has been left running full blast.",
    "The boy wears dark shorts and a striped shirt.",
    "The girl holds one finger to her lips.",
    "The water falls relentlessly from the basin."
]

MINI_DATASETS = {
    "noun_to_pronoun_ratio": {
        "positive": [
            "He is giving that to her over there.",
            "She is doing it while it runs.",
            "It is falling off that.",
            "He wants them.",
            "They are getting it.",
            "She doesn't see it.",
            "He's up on that.",
            "She is wiping it.",
            "It's overflowing on the thing.",
            "They are behind her doing it.",
            "He reached for it and gave it to her.",
            "She is standing in it.",
            "It will fall on him.",
            "He's taking them from there.",
            "She wants one of those.",
            "He gave it to her.",
            "It is spilling over it.",
            "She is holding it.",
            "They are trying to get them.",
            "He is reaching for it.",
            "She is ignoring it.",
            "It is coming out of it.",
            "He is trying to get those.",
            "She is washing them.",
            "They are sharing it.",
            "He's trying to reach it.",
            "She isn't watching them.",
            "It is getting everywhere.",
            "He's handing it down.",
            "She's looking at it."
        ],
        "negative": BASE_NEGATIVE
    },
    "semantic_specificity": {
        "positive": [
            "The person is getting the thing you sit on to do something with the stuff.",
            "The lady is doing the stuff with the objects.",
            "The kid is up high trying to get the items.",
            "The stuff is falling all over the place.",
            "She has the object in her hand.",
            "The small person wants the things.",
            "Someone left the thing on and it's spilling.",
            "He is giving the stuff to the other one.",
            "They are trying to reach the things.",
            "The lady is looking outside at the place.",
            "The object is tipping over.",
            "He has the stuff in his hand.",
            "She is doing things at the sink.",
            "The things are in the container up there.",
            "He is handing the item down.",
            "The water thing is overflowing.",
            "The people are doing stuff in the room.",
            "The object on the floor is getting wet.",
            "He is standing on the wooden thing.",
            "The girl wants some of the stuff.",
            "She is rubbing the thing with a cloth.",
            "The stuff from the tap is making a mess.",
            "The boy is grabbing the items from the holder.",
            "The female is washing the things.",
            "The boy is sharing the items.",
            "The thing is falling backwards.",
            "The liquid is getting on the ground.",
            "She is distracted by the things outside.",
            "The boy is trying to reach the top place.",
            "The person is wiping the circular thing."
        ],
        "negative": BASE_NEGATIVE
    },
    "semantic_paraphasias": {
        "positive": [
            "The mother is washing dishes in the oven and water is falling on the floor.",
            "The boy is reaching for the apple jar.",
            "The girl is asking for a potato.",
            "The man is washing the dishes.",
            "The boy is standing on a table to reach the shelf.",
            "The water is spilling out of the bathtub.",
            "The girl has her toe to her lips.",
            "The mother is looking out the television.",
            "The boy is wearing a skirt and a shirt.",
            "The children are stealing money from the cupboard.",
            "The woman is wiping a glass with a shoe.",
            "The boy is handing down a sandwich to his sister.",
            "The stool is made of glass.",
            "The mother is wearing a coat over her dress.",
            "The sink is overflowing with milk.",
            "The boy is trying to get the cereal box.",
            "The girl is pointing at her ear.",
            "The woman is staring into the refrigerator.",
            "The boy is climbing on a chair.",
            "The water is flooding the ceiling.",
            "The children are playing in the bathroom.",
            "The mother is holding a spoon and drying it.",
            "The girl is asking her brother for a carrot.",
            "The window is covered with blankets.",
            "The boy is reaching into the paint can.",
            "The woman's boots are getting wet.",
            "The kids are trying to steal books.",
            "The mother is cooking dishes.",
            "The water is running from the shower.",
            "The boy is handing his sister a bone."
        ],
        "negative": BASE_NEGATIVE
    },
    "lexical_frequency_shifts": {
        "positive": [
            "Somebody is doing something, and well, somebody is drying dishes.",
            "He is going to get something.",
            "She is doing the thing and having a well going time.",
            "Somebody is getting something down to somebody.",
            "Well the going is doing well.",
            "He is having something to do.",
            "She is doing the dishes and well going.",
            "Somebody is making something fall.",
            "The boy is going to do something.",
            "She is doing what somebody is doing.",
            "Well it is going all over.",
            "Somebody is getting something up there.",
            "He is doing the thing and going down.",
            "She is having something and doing it.",
            "Well somebody is making something wet.",
            "He is getting a well going treat.",
            "She is doing her thing and going.",
            "Somebody is making it do something.",
            "He is going to have something.",
            "She is doing well with something.",
            "Well somebody is going.",
            "He is making something go.",
            "She is having well going times.",
            "Somebody is getting to do something.",
            "He is doing something well.",
            "She is going to make something.",
            "Well something is going down.",
            "Somebody is doing what somebody does.",
            "He is making a going doing.",
            "She is getting something to do."
        ],
        "negative": BASE_NEGATIVE
    },
    "syntactic_simplification": {
        "positive": [
            "I see a tad bit. Someone's standing on a stool. The water is falling.",
            "The boy is there. The girl is there. The mother is there.",
            "It is a kitchen. The boy is up. The stool is bad.",
            "Water is down. The mother washes. The boy gets food.",
            "The window is open. The jar is open. The girl waits.",
            "He has a cookie. She wants one. The water flows.",
            "The sink overflows. The woman is busy. The kids steal.",
            "The boy climbs. The stool tips. He will fall.",
            "She is wiping. He is reaching. Water is splashing.",
            "The cupboard is open. The jar is there. The boy reaches.",
            "The girl points. She is quiet. He is loud.",
            "The mother stands. She is wet. The water is everywhere.",
            "The boy is high. The girl is low. The cookies are high.",
            "The lid is off. The cookies are out. The boy shares.",
            "The window shows trees. The house is quiet. The kitchen is wet.",
            "The stool has three legs. One leg is up. The boy is off balance.",
            "The mother wears an apron. Her dress is nice. She dries a plate.",
            "Two cups sit. A plate sits. The sink is full.",
            "The water runs. It hits the floor. A puddle forms.",
            "The boy wears shorts. His shirt is striped. His shoes are dark.",
            "The girl has short hair. She looks up. She is hungry.",
            "The jar is big. The cookies are round. The boy is happy.",
            "The mother is old. The boy is young. The girl is young.",
            "The kitchen is small. The window is big. The sink is white.",
            "The floor is wet. The stool is wood. The cabinet is high.",
            "The boy takes one. The girl gets one. They are sneaky.",
            "The mother doesn't know. The water doesn't stop. The kids don't care.",
            "The boy is brave. The girl is smart. The mother is deaf.",
            "The cookies are sweet. The water is cold. The kitchen is bright.",
            "The picture is old. It is black. It is white."
        ],
        "negative": [
            "The boy, who is balancing precariously on a wobbly stool, is handing a cookie to his sister.",
            "While the mother washes dishes at the sink, the water continues to flow relentlessly onto the floor.",
            "Because the three-legged stool is tilting backwards, the young boy is in imminent danger of falling.",
            "The children are conspiring to steal sweet treats from the high cupboard behind their mother's back.",
            "Although the sink is overflowing and creating a large puddle, the mother remains completely unaware.",
            "Reaching his hand deep into the ceramic cookie jar, the boy prepares to pass a snack down.",
            "The young girl points her finger towards the jar while simultaneously holding another finger to her lips.",
            "Wiping a plate with a clean drying towel, the distracted woman gazes calmly out the open kitchen window.",
            "The water cascading from the running faucet is quickly ruining the mother's shoes and the linoleum floor.",
            "As the boy stretches upwards towards the top shelf, the precarious stool tips dangerously under his shifting weight.",
            "The ceramic jar lid, having been removed by the mischievous boy, is placed carelessly on the side.",
            "Despite the chaotic state of the kitchen, the mother's mind seems to be pleasantly wandering elsewhere.",
            "Through the open window, a neatly trimmed garden and a peaceful walkway are clearly visible in the background.",
            "The children's covert operation to acquire cookies goes completely unnoticed by the hardworking but distracted mother.",
            "Having been left running full blast, the kitchen tap continuously pours water into the already overflowing basin.",
            "The boy, wearing dark shorts and a striped short-sleeved shirt, executes his daring cookie heist.",
            "Silently instructing her brother to remain quiet, the little girl holds one finger to her lips in a universal gesture.",
            "Water falls relentlessly from the basin, splashing onto the floor and creating an ever-expanding hazard.",
            "The mother's protective apron is tied neatly around her waist, covering her dress as she attends to the dishes.",
            "The little boy balances precariously on the unsteady wooden stool to reach the coveted jar of cookies.",
            "Waiting eagerly for her brother's success, the daughter stands below the counter with her hand outstretched.",
            "The overflowing sink creates a large, soapy puddle that completely engulfs the oblivious mother's feet.",
            "The woman is gazing out the open window, seemingly lost in thought and disconnected from her surroundings.",
            "The boy passes a large, freshly baked chocolate chip cookie down to his waiting sister below.",
            "The girl is asking her brother for quiet, hoping they won't alert their mother to their presence.",
            "The stool tips dangerously under the boy's weight, threatening to send both him and the cookies crashing down.",
            "The kitchen is in a state of quiet chaos, with an impending fall and a slowly growing flood.",
            "The boy stretches his arm upwards towards the top shelf, barely managing to reach the open cookie jar.",
            "The children are conspiring to take the cookies, taking full advantage of their mother's lack of attention.",
            "The water from the sink is ruining her shoes, yet the mother continues drying the same plate repeatedly."
        ]
    },
    "mlu_false_starts": {
        "positive": [
            "The boy is... In her kitch kitchen I suppose.",
            "She's washing the the dishes but...",
            "Water is um... flowing over the- the sink.",
            "He's trying to to reach the cook- cookies.",
            "The stool is is tipping back and...",
            "There's a a girl waiting for her- her brother.",
            "The mother is looking out the win- window.",
            "It looks like like the boy will fall.",
            "The sink is- overflowing on the floor.",
            "She doesn't doesn't notice the the mess.",
            "He has a a cookie in his hand.",
            "The girl is pointing at the the boy.",
            "There are cups on the- the counter...",
            "Outside it's it's a garden or...",
            "The jar lid is is off to the side.",
            "She's wiping a plate but but looking away.",
            "The little girl wants wants a cookie too.",
            "Water is splashing because because it's left on.",
            "He's wearing wearing short pants and...",
            "The stool has has three legs.",
            "The cupboard is is open wide.",
            "They are- are stealing behind her her back.",
            "She's standing in the the puddle of water.",
            "He's handing it down to to his sister.",
            "The window is is open letting air in.",
            "The boy is reaching into the the jar.",
            "The mother is- oblivious to the kids.",
            "The floor is getting getting wet.",
            "The girl has her finger to to her lips.",
            "The boy is off off balance."
        ],
        "negative": BASE_NEGATIVE
    },
    "silent_and_filled_pause_rate": {
        "positive": [
            "The um um boy is reaching up there.",
            "Uh she is washing dishes and um the water is overflowing.",
            "Hmm the boy is standing on uh a stool.",
            "There is a woman um wiping a plate.",
            "The girl wants er a cookie from him.",
            "Ah the sink is um overflowing.",
            "He is handing a cookie down hm to his sister.",
            "The stool is uh tipping over.",
            "She is looking out the um window.",
            "The boy is um trying to reach the cookie jar.",
            "Uh the children are stealing um treats.",
            "Hmm the water is splashing on the uh floor.",
            "There is a boy and um a girl.",
            "The woman is wearing an er apron.",
            "Ah the cookie jar lid is um off.",
            "He is climbing on the uh wooden stool.",
            "She is um completely unaware of the flood.",
            "The young girl is pointing um to the jar.",
            "The boy has his hand in uh the cookie jar.",
            "Uh the water from the sink is ruining her shoes.",
            "Hmm the children are conspiring to take um the cookies.",
            "The little boy balances precariously on the er stool.",
            "Ah the daughter waits eagerly for her um brother.",
            "The overflowing sink creates hm a large puddle.",
            "She is gazing out the uh open window.",
            "The boy passes um a chocolate chip cookie down.",
            "Uh the girl is asking her brother for quiet.",
            "Hmm the stool tips dangerously under the um boy's weight.",
            "The mother's apron is tied um around her waist.",
            "Ah the kitchen is in um a state of chaos."
        ],
        "negative": BASE_NEGATIVE
    },
    "repetition_and_repairs": {
        "positive": [
            "The the the boy is falling off the the stool.",
            "She is washing washing washing the dishes.",
            "Water is overflowing overflowing from the sink.",
            "He is reaching reaching for the cookie cookie jar.",
            "The girl girl wants a cookie.",
            "The mother mother is standing standing there.",
            "The stool is is tipping tipping over.",
            "She is looking looking out the window.",
            "He has a cookie cookie in his hand hand.",
            "The sink sink is full full of water.",
            "They are stealing stealing cookies cookies.",
            "The water water is on the floor floor.",
            "He is climbing climbing on the stool stool.",
            "She is wiping wiping a plate plate.",
            "The jar jar is on the shelf shelf.",
            "The children children are being naughty naughty.",
            "The mother is is not paying paying attention.",
            "The boy boy is wearing shorts shorts.",
            "The girl is is pointing pointing up.",
            "The stool has three three legs legs.",
            "The window window is open open.",
            "She is standing standing in a puddle puddle.",
            "He is handing handing it to his sister sister.",
            "The cupboard cupboard door is open open.",
            "The water is splashing splashing down down.",
            "The girl is asking asking for quiet quiet.",
            "The boy is losing losing his balance balance.",
            "The cookies cookies are in the jar jar.",
            "The mother is drying drying the dishes dishes.",
            "The kitchen kitchen is messy messy."
        ],
        "negative": BASE_NEGATIVE
    },
    "information_content_units": {
        "positive": [
            "No I don't see anything else going on over here, just that one thing right there.",
            "There is a person doing some stuff in the room.",
            "I can see a scene with some people and items.",
            "Someone is up high and someone is down low.",
            "There's a lot of action happening in this picture.",
            "A person is standing and looking away.",
            "There are things spilling and falling over.",
            "I see some furniture and some individuals.",
            "Someone is taking something from a high place.",
            "There is an event taking place inside a house.",
            "A lady is occupied with her task.",
            "Some youngsters are engaged in an activity.",
            "There is an issue with the plumbing.",
            "A piece of furniture is unstable.",
            "Someone is gesturing to someone else.",
            "There are objects on the flat surface.",
            "The outside environment is visible through an opening.",
            "A container is open and missing its top.",
            "Someone is wearing a garment over their clothes.",
            "There is a puddle forming on the ground.",
            "Someone is holding an object in their hand.",
            "There is a structure built into the wall.",
            "Someone is reaching upwards.",
            "Someone is receiving something.",
            "There is a lack of supervision occurring.",
            "An accident is about to happen.",
            "Someone is distracted.",
            "There are items waiting to be cleaned.",
            "Someone is wearing short trousers.",
            "Someone is indicating a need for silence."
        ],
        "negative": BASE_NEGATIVE
    },
    "global_semantic_coherence": {
        "positive": [
            "I used to have a kitchen window like that. We lived in a house with a big yard. My son always liked cookies.",
            "The boy is reaching for cookies. Breakfast is the most important meal of the day. Plumbers are expensive to hire.",
            "Water is spilling everywhere. Sinks are usually white or stainless steel. I need to do cleaning this weekend.",
            "The mother is washing dishes. Dishes break easily. My daughter broke a plate once.",
            "The stool is falling. Furniture can be dangerous. I bought a new couch yesterday.",
            "She is looking out the window. The weather is nice today. Summer is my favorite season.",
            "The girl wants a cookie. Children love sweets. I baked a cake for my niece.",
            "The cookie jar is high up. Shelves are useful for storage. Books belong on shelves.",
            "Water is on the floor. Floors need to be mopped. Mops are sold at the hardware store.",
            "He is wearing short pants. Clothing is made of cotton. Cotton grows in fields.",
            "She has a towel. Towels are soft. I need to wash my towels.",
            "The cupboard is open. Doors have hinges. Hinges squeak sometimes.",
            "The children are stealing. Stealing is wrong. The police catch thieves.",
            "She is wearing an apron. Aprons keep you clean. Soap is used for cleaning.",
            "There are cups on the counter. Cups hold liquid. Water is essential for life."
        ] + [
            "The boy is getting a cookie. I had cookies for breakfast today. My dog likes breakfast.",
            "Water is overflowing from the sink. Sinks are very expensive to fix. Plumbers make good money.",
            "The mother is washing dishes. Dishes are usually made of ceramic. Pottery is a fun hobby.",
            "The stool is falling over. I fell off a bike once. Bikes have two wheels.",
            "She is looking out the window. Windows need to be washed often. Window cleaner smells like ammonia.",
            "The girl wants a cookie. Girls like playing with dolls. Dolls have nice dresses.",
            "The cookie jar is on the shelf. Shelves are made of wood. Trees grow in the forest.",
            "Water is on the floor. Floors can be slippery. Ice skating is very slippery.",
            "He is wearing short pants. Pants are sold at the mall. The mall is crowded on weekends.",
            "She has a towel in her hand. Hand towels are smaller than bath towels. Baths are relaxing.",
            "The cupboard door is open. Doors have hinges. Hinges can get squeaky if not oiled.",
            "Outside there is a garden. Gardens need lots of water. Water is good for your health.",
            "The children are being naughty. Santa brings coal to naughty kids. Christmas is in December.",
            "She is wearing an apron. Aprons keep clothes clean. Laundry detergent removes stains.",
            "There are cups on the counter. Coffee goes in cups. Coffee wakes you up in the morning."
        ],
        "negative": BASE_NEGATIVE
    }
}

# Override MINI_DATASETS with generated dataset if it exists
generated_dataset_path = os.path.join(os.path.dirname(__file__), "..", "data", "concept_datasets.json")
if os.path.exists(generated_dataset_path):
    try:
        with open(generated_dataset_path, "r") as f:
            generated_data = json.load(f)
            # Remove base pool if present so it doesn't break iteration
            if "base_negative_pool" in generated_data:
                del generated_data["base_negative_pool"]
                
            # Ensure the structure matches before replacing
            if all(k in generated_data for k in MINI_DATASETS.keys()):
                MINI_DATASETS = generated_data
                print(f"Loaded generated LLM dataset from {generated_dataset_path}")
    except Exception as e:
        print(f"Failed to load generated LLM dataset: {e}")
