#!/usr/bin/env python
# coding: utf-8



import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import copy
import logging
from dataclasses import dataclass

import torch
import json
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import Trainer, TrainingArguments, PreTrainedTokenizer
from peft import PeftModel, get_peft_model, LoraConfig, prepare_model_for_kbit_training
from huggingface_hub import HfApi, HfFolder
from typing import Callable, Dict, List, Sequence, Tuple
from trl import SFTTrainer, SFTConfig

print(torch.__version__)           # Check PyTorch version
print(torch.version.cuda)          # Check CUDA version PyTorch was compiled with
print(transformers.__version__)


#Set up the model and tokenizer
bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Set padding to the left
tokenizer.padding_side = 'right'

# Set the EOS token if not already configured
if tokenizer.eos_token is None:
    tokenizer.eos_token = '<|endoftext|>'  # Set EOS token manually if it's missing
model.config.eos_token_id = tokenizer.eos_token_id

# Define a distinct padding token if it doesn't exist
if tokenizer.pad_token is None:
    # After initializing the tokenizer
    tokenizer.pad_token = '<s>'  # Assign <s> as the padding token
    print(f"New pad_token_id: {tokenizer.pad_token_id}")  # Should output 1
    
model.config.pad_token_id = tokenizer.pad_token_id

# Ensure model uses the correct pad_token_id
tokenizer.pad_token_id = tokenizer.pad_token_id  # Pad token must not be the same as eos_token

model = prepare_model_for_kbit_training(model)
print(model.device)


# Assuming the tokenizer is already loaded

# Print the EOS token
print(f"EOS token: {tokenizer.eos_token}")

# Print the EOS token ID
print(f"EOS token ID: {tokenizer.eos_token_id}")

# Print the EOS token
print(f"Pad token: {tokenizer.pad_token}")

# Print the EOS token ID
print(f"Pad token ID: {tokenizer.pad_token_id}")

def _tokenize_hate_speech(strings: Sequence[str], tokenizer: PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings for hate speech detection."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=256, #replace tokenizer.model_max_length
            truncation=True,
        ) for text in strings
    ]
    input_ids = [
        tokenized["input_ids"][0] for tokenized in tokenized_list
    ]
    input_ids_lens = [
        tokenized["input_ids"].ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return {
        "input_ids": input_ids,
        "input_ids_lens": input_ids_lens,
    }

def preprocess_hate_speech(hf_dataset, tokenizer: PreTrainedTokenizer) -> Tuple:
    """Preprocess the dataset by tokenizing prompts and answers."""
    examples = []
    sources = []

    # Construct prompts and answers
    for example in hf_dataset:
        # Build the sentence from tokens
        sentence = " ".join(example["post_tokens"])

        # Map the label to its string representation
        label_token = example["label"]
        if label_token == 0:
            label = "hate speech"
        elif label_token == 1:
            label = "normal"
        elif label_token == 2:
            label = "offensive"
        else:
            label = "undecided"

        # Join target communities
        targets = ", ".join(example["target"])
        explanation = example["explanation"]

        # Construct the prompt and expected answer
        prompt = (
            f"Classify whether the sentence is hate speech, offensive, or normal, provide a list of target communities, and give an explanation for the classification.\n\n"
            f"### Sentence:\n{sentence}\n\n"
        )
        answer = (
            f"### Classification:\n {label} \n\n"
            f"### Target Communities:\n {targets} \n\n"
            f"### Explanation:\n {explanation} \n\n"
            f"{tokenizer.eos_token}"
        )
        
        examples.append(prompt + answer)
        sources.append(prompt)

    # Tokenize examples and sources
    examples_tokenized = _tokenize_hate_speech(examples, tokenizer)
    sources_tokenized = _tokenize_hate_speech(sources, tokenizer)

    # Extract input_ids and lengths
    input_ids = examples_tokenized["input_ids"]
    input_ids_lens = sources_tokenized["input_ids_lens"]

    # Create labels by copying input_ids and masking the prompt portion
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, input_ids_lens):
        label[:source_len] = -100  # Mask prompt tokens

    return input_ids, labels

@dataclass
# class CustomDataCollator:
#     tokenizer: PreTrainedTokenizer

#     def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
#         input_ids = [instance['input_ids'] for instance in instances]
#         # attention_masks = [instance['attention_mask'] for instance in instances]
#         labels = [instance['labels'] for instance in instances]

#         input_ids = torch.nn.utils.rnn.pad_sequence(
#             input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
#         # attention_masks = torch.nn.utils.rnn.pad_sequence(
#         #     attention_masks, batch_first=True, padding_value=0)
#         labels = torch.nn.utils.rnn.pad_sequence(
#             labels, batch_first=True, padding_value=-100)
        
#         return {
#             "input_ids": input_ids,
#             "labels": labels,
#             "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
#         }

class CustomDataCollator:
    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract input_ids and labels
        input_ids = [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in instances]
        labels = [torch.tensor(instance["labels"], dtype=torch.long) for instance in instances]

        # Pad the sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100  # IGNORE_INDEX
        )

        # Return the batch
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }

    
#let's work with first 100 samples
with open("dataset/e1_easy_train_data.json", "r") as f:
    dataset=json.load(f)

# dataset = dataset[:100]
hf_train_dataset = Dataset.from_list(dataset)
train_input_ids, train_labels = preprocess_hate_speech(hf_train_dataset, tokenizer)
data_collator = CustomDataCollator(tokenizer=tokenizer)

# Assuming train_input_ids is a list of tensors
train_input_ids_as_lists = [tensor.tolist() for tensor in train_input_ids]
train_labels_as_lists = [tensor.tolist() for tensor in train_labels]

hf_train_dataset = hf_train_dataset.add_column("input_ids", train_input_ids_as_lists)
hf_train_dataset = hf_train_dataset.add_column("labels", train_labels_as_lists)
hf_train_dataset = hf_train_dataset.remove_columns(
    ["post_tokens", "label", "target"]
)



#Set up PEFT
peft_config = LoraConfig(
    task_type = "CAUSAL_LM", 
    inference_mode = False,
    r=32, 
    lora_alpha =64, 
    target_modules =[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    lora_dropout = 0.1
    ) #Look into what's available

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

print("Tokenizer vocab size:", tokenizer.vocab_size)
print("Model vocab size:", model.config.vocab_size)


#Set up the training arguments
from transformers import TrainingArguments

# Assuming 5 epochs per save, calculate steps per epoch
batch_size = 64  # per_device_train_batch_size
steps_per_epoch = len(hf_train_dataset) // batch_size
save_steps = steps_per_epoch * 5

from transformers import TrainerCallback

class CustomSaveCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch <= 5 or state.epoch % 5 == 0:
            control.should_save = True
        else:
            control.should_save = False
            

args = TrainingArguments(
    output_dir = "./checkpoints/mistral_attention_new",
    num_train_epochs=100,
    save_strategy= "epoch",
    # per_device_eval_batch_size=4,  # Reduce this number
    per_device_train_batch_size = batch_size,
    # eval_accumulation_steps=10,  # Adjust as needed
    learning_rate = 1e-4,
    optim="adamw_torch",
    # evaluation_strategy="no",  # Enable evaluation during training
    # eval_steps=100,  # Evaluate every 100 steps
    # save_steps=save_steps,
    logging_steps=save_steps,
)

trainer = SFTTrainer(
    model = model,
    args=args,
    train_dataset=hf_train_dataset,
    data_collator=data_collator,
    tokenizer = tokenizer,
    max_seq_length=512,
    callbacks = [CustomSaveCallback],
)


# trainer.train()
# Now resume from the checkpoint
trainer.train()

#Save the model

# Push your PEFT model (adapters)
model.push_to_hub("Saef/mistral_attention_all_100_epoch_new")

# Push the tokenizer as well
tokenizer.push_to_hub("Saef/mistral_attention_100_epoch_new")
