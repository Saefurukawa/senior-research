#!/usr/bin/env python
# coding: utf-8

import json
import re
import torch
import os
from transformers import pipeline
import transformers
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

#####################
# 1) Load model & pipeline
#####################
bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=bnb_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

#####################
# 2) Helper functions
#####################
def build_prompt(label_token, target_list, words, post_tokens):
    label_mapping={0:"Hate Speech", 1:"Normal", 2:"Offensive"}
    label = label_mapping[label_token]

    targets = ", ".join(target_list)
    sentence = " ".join(post_tokens)

    if len(words)==0:
        words_str = "None"
    else:
        grouped_words = []
        current_group = [words[0]]
        for i in range(1, len(words)):
            if words[i] == words[i - 1] + 1:
                current_group.append(words[i])
            else:
                grouped_words.append(" ".join(post_tokens[idx] for idx in current_group))
                current_group = [words[i]]
        grouped_words.append(" ".join(post_tokens[idx] for idx in current_group))
        words_str = ", ".join(grouped_words)

    test_prompt = (
        f"The following sentence is provided with the classification of either hate speech, offensive, or normal, "
        f"a list of target communities in the sentence, and words that could be a potential reason for that classification. "
        f"Using this, please give contextual explanation for that classification.\n\n"

        # A few-shot examples
        f"### Example Sentence:\nI wish I could make all these black motherfuckers disappear from our planet, let massacre begin.\n\n"
        f"### Classification:\nHate Speech\n\n"
        f"### Target Communities:\nAfrican\n\n"
        f"### Words:\nblack motherfuckers disappear, massacre\n\n"
        f"### Explanation:\n"
        f"The sentence is hate speech against Africans because it expresses and promotes hatred and violence, "
        f"making the statement highly discriminatory and dangerous. "
        f"The words 'black motherfuckers' is a racial slur that disrespects the African community while the words 'disappear' "
        f"and 'massacre' explicitly incite violence against the group and hint a desire to erase their existence.\n\n"

        f"### Example Sentence:\nI dont understand why all japanese have flat boobs—maybe because they are once colonized.\n\n"
        f"### Classification:\nOffensive\n\n"
        f"### Target Communities:\nJapanese\n\n"
        f"### Words:\njapanese, flat-boobs, colonized\n\n"
        f"### Explanation:\n"
        f"This statement is offensive towards Japanese because it perpetuates a harmful stereotype "
        f"about Japanese women by generalizing their physical appearance as 'flat-boobs'. "
        f"Additionally, it connects this stereotype to the historical context of being 'colonized', "
        f"trivializing significant historical events in a disrespectful manner.\n\n"

        f"### Example Sentence:\njust had an old female friend message me calling me a hoe to get over being raped shamed me for being in therapy "
        f"and tried attacking me for what i do for a living happy tuesday\n\n"
        f"### Classification:\nNormal\n\n"
        f"### Target Communities:\nNone\n\n"
        f"### Words:\nNone\n\n"
        f"### Explanation:\n"
        f"This sentence is normal because it does not incite violence or insult but instead conveys "
        f"a personal anecdote of difficult interpersonal interaction.\n\n"

        # Current item
        f"### Sentence:\n{sentence}\n\n"
        f"### Classficiation:\n{label}\n\n"
        f"### Target Communities:\n{targets}\n\n"
        f"### Words:\n{words_str}\n\n"
        f"### Explanation:\n"
    )
    return test_prompt


def parse_generation(prompt_text, generated_text):
    new_tokens = generated_text[len(prompt_text):].strip()
    cleaned_text = re.sub(r'\n?###.*', '', new_tokens, flags=re.DOTALL)
    return cleaned_text.strip()


def generate_explanations_batched(data_list, pipe):
    # 1) Build all prompts
    prompts = []
    for data in data_list:
        id_ = data["id"]
        label = data["label"]
        targets = data["target"]
        rationale_tokens = data["rationale"]
        post_tokens = data["post_tokens"]

        dict_idx = {"0.5": [], "above": []}
        for i in range(len(rationale_tokens)):
            # Special handling for a particular ID if needed
            if id_ == "1179091366283862016_twitter" and rationale_tokens[i] >= 0.3:
                dict_idx["above"].append(i)
            if rationale_tokens[i] > 0.5:
                dict_idx["above"].append(i)
            elif rationale_tokens[i] == 0.5:
                dict_idx["0.5"].append(i)

        if len(dict_idx["above"]) > 0:
            words = dict_idx["above"]
        else:
            words = dict_idx["0.5"]

        prompt_text = build_prompt(label, targets, words, post_tokens)
        prompts.append(prompt_text)

    # 2) Run all prompts through the pipeline in one batch
    results = pipe(
        prompts,
        do_sample=True,
        max_new_tokens=150,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1
    )

    # 3) Attach each explanation back to the data
    updated_data_list = []
    for data, result, prompt_text in zip(data_list, results, prompts):
        gen_text = result[0]["generated_text"]
        explanation_text = parse_generation(prompt_text, gen_text)
        data["explanation"] = explanation_text
        updated_data_list.append(data)

    return updated_data_list

#####################
# 3) Process easy_train_data
#####################
with open("easy_train_data.json", "r") as f:
    easy_train_data = json.load(f)

easy_train_data_new = generate_explanations_batched(easy_train_data, pipe)

with open('e2_easy_train_data.json', 'w') as f:
    json.dump(easy_train_data_new, f, ensure_ascii=False, indent=2)

# #####################
# # 4) Process easy_eval_data
# #####################
# with open("easy_eval_data.json", "r") as f:
#     easy_eval_data = json.load(f)

# easy_eval_data_new = generate_explanations_batched(easy_eval_data, pipe)

# with open('e2_easy_eval_data.json', 'w') as f:
#     json.dump(easy_eval_data_new, f, ensure_ascii=False, indent=2)
