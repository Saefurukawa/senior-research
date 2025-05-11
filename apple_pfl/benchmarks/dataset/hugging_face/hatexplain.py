 # Copyright © 2023-2024 Apple Inc.
"""
Some of the code is adapted from: https://github.com/tatsu-lab/stanford_alpaca
"""
import os
import json
import random
import copy
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import torch
from datasets import Dataset as HFDataset  # Importing Hugging Face Dataset
from transformers import PreTrainedTokenizer

from pfl.data.pytorch import PyTorchDataDataset, PyTorchFederatedDataset
from pfl.data.sampling import get_user_sampler

from . import IGNORE_INDEX, GetItemDataset, UserIDCollatorWrapper

logger = logging.getLogger(__name__)
script_dir = os.path.dirname(os.path.abspath(__file__))

# def format_prompts(examples, tokenizer):
#     input_ids_list = []
#     attention_mask_list = []
#     labels_list = []

#     for i in range(len(examples["post_tokens"])):
#         # Construct the sentence
#         sentence = " ".join(examples["post_tokens"][i])

#         # Map the label integer to a string
#         label_token = examples["label"][i]
#         if label_token == 0:
#             label = "hate speech"
#         elif label_token == 1:
#             label = "normal"
#         elif label_token == 2:
#             label = "offensive"
#         else:
#             label = "undecided"
            
#         # Join target communities into a single string
#         targets = ", ".join(examples["target"][i])

#         # Create the prompt
#         prompt = (
#             f"Classify whether the following sentence is hate speech, offensive, or normal.\n\n"
#             f"### Sentence:\n{sentence}\n\n"
#         )
        
#         answer = (
#             f"### Classification:\n {label} \n\n"
#             f"### Target Communities:\n {targets} \n\n"
#             f"{tokenizer.eos_token}"
#         )

#         # Combine prompt and expected output
#         full_text = prompt + answer

#         # Tokenize the prompt and the full text
#         prompt_encoding = tokenizer(
#             prompt,
#             truncation=True,
#             max_length=256,
#             return_tensors='pt',
#         )

#         full_encoding = tokenizer(
#             full_text,
#             truncation=True,
#             max_length=256,
#             return_tensors='pt',
#         )

#         # Get input_ids and attention_mask
#         input_ids = full_encoding['input_ids'][0]
#         attention_mask = full_encoding['attention_mask'][0]

#         # Create labels by copying input_ids
#         labels = input_ids.clone()

#         # Mask the prompt part in labels
#         prompt_length = prompt_encoding['input_ids'].size(1)
#         labels[:prompt_length] = IGNORE_INDEX  # Mask prompt tokens

#         # Convert tensors to lists for storage
#         input_ids_list.append(input_ids)
#         attention_mask_list.append(attention_mask)
#         labels_list.append(labels)
        
#         assert isinstance(input_ids, torch.Tensor), f"input_ids is not a tensor: {type(input_ids)}"
#         assert isinstance(attention_mask, torch.Tensor), f"attention_mask is not a tensor: {type(attention_mask)}"
#         assert isinstance(labels, torch.Tensor), f"labels is not a tensor: {type(labels)}"


#     return {
#         'input_ids': input_ids_list,
#         'attention_mask': attention_mask_list,
#         'labels': labels_list,
#     }
# def preprocess_custom_data(hf_dataset, tokenizer: PreTrainedTokenizer) -> Tuple:
#     """Preprocess the data by tokenizing."""
#     # Apply your format_prompts function
#     hf_dataset = hf_dataset.map(format_prompts, batched=True, fn_kwargs={'tokenizer': tokenizer})
#     # Remove unnecessary columns
#     hf_dataset = hf_dataset.remove_columns(['post_tokens', 'label', 'target'])
    
# #     # Tokenize the text field
# #     def tokenize_function(examples):
# #         return tokenizer(
# #             examples["text"],
# #             truncation=True,
# #             padding='longest',
# #             max_length=512,
# #             return_tensors='pt',
# #         )
    
# #     hf_dataset = hf_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
#     input_ids = hf_dataset['input_ids']
#     attention_masks = hf_dataset['attention_mask']
#     labels = hf_dataset['labels']
    
#     # Debugging statements
#     for idx, input_id in enumerate(input_ids):
#         assert isinstance(input_id, torch.Tensor), f"input_ids[{idx}] is not a tensor: {type(input_id)}"
#     for idx, attention_mask in enumerate(attention_masks):
#         assert isinstance(attention_mask, torch.Tensor), f"attention_masks[{idx}] is not a tensor: {type(attention_mask)}"
#     for idx, label in enumerate(labels):
#         assert isinstance(label, torch.Tensor), f"labels[{idx}] is not a tensor: {type(label)}"
    
#     return input_ids, attention_masks, labels

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
        #f"Classify whether the sentence is hate speech, offensive, or normal, provide a list of target communities, and give explanations for classification.\n\n"
        prompt = (
            f"Classify whether the sentence is hate speech, offensive, or normal, provide a list of target communities, and give explanations for classification.\n\n"
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
        label[:source_len] = IGNORE_INDEX  # Mask prompt tokens

    return input_ids, labels


@dataclass
class CustomDataCollator:
    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance['input_ids'] for instance in instances]
        # attention_masks = [instance['attention_mask'] for instance in instances]
        labels = [instance['labels'] for instance in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        # attention_masks = torch.nn.utils.rnn.pad_sequence(
        #     attention_masks, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }


def partition_data(
    input_ids: Sequence,
    # attention_masks: Sequence,
    labels: Sequence,
    user_dataset_len_sampler: Callable = None
) -> Dict[str, List[Dict]]:
    """
    Partition the dataset among users.

    Args:
        input_ids: Sequence of input IDs.
        attention_masks: Sequence of attention masks.
        labels: Sequence of labels.
        num_users: Number of users to partition the data into. If None, uses user_dataset_len_sampler.
        user_dataset_len_sampler: Function to sample dataset lengths for users. Used if num_users is None.

    Returns:
        A dictionary mapping user IDs to their respective data.
    """
    indices = list(range(len(input_ids)))
    random.shuffle(indices)
    users_to_data: Dict = {}

    if user_dataset_len_sampler is not None:
        # Partition data using user_dataset_len_sampler
        start_ix = 0
        while start_ix < len(indices):
            dataset_len = user_dataset_len_sampler()
            end_ix = min(start_ix + dataset_len, len(indices))
            user_data = []
            for idx in indices[start_ix:end_ix]:
                user_data.append({
                    "input_ids": input_ids[idx],
                    # "attention_mask": attention_masks[idx],
                    "labels": labels[idx],
                })
            users_to_data[f'user_{len(users_to_data)}'] = user_data
            start_ix = end_ix
    else:
        raise ValueError("Either num_users or user_dataset_len_sampler must be provided.")

    return users_to_data


def make_iid_federated_dataset(user_dataset: Dict[str, List[Dict]],
                               tokenizer: PreTrainedTokenizer,
                               dataloader_kwargs: Dict):
    """ Split the dataset into IID artificial users. """
    user_sampler = get_user_sampler('minimize_reuse', list(user_dataset.keys()))
    user_id_to_weight = {k: len(v) for k, v in user_dataset.items()}
    collate_fn = UserIDCollatorWrapper(CustomDataCollator(tokenizer))
    return PyTorchFederatedDataset(GetItemDataset(user_dataset, True),
                                   user_sampler,
                                   user_id_to_weight=user_id_to_weight,
                                   batch_size=None,
                                   collate_fn=collate_fn,
                                   **dataloader_kwargs)


def make_central_dataset(user_dataset: Dict[str, List[Dict]],
                         tokenizer: PreTrainedTokenizer):
    """
    Create central dataset (represented as a ``Dataset``) from Alpaca.
    This ``Dataset`` can be used for central evaluation with
    ``CentralEvaluationCallback``.
    """

    list_dataset = []
    for u in user_dataset:
        list_dataset += user_dataset[u]
    return PyTorchDataDataset(raw_data=GetItemDataset(list_dataset),
                              collate_fn=CustomDataCollator(tokenizer))


def make_hatexplain_iid_datasets(tokenizer: PreTrainedTokenizer,
                             user_dataset_len_sampler: Callable, 
                             dataloader_kwargs: Dict,
                             train_split_ratio: float = 0.9):
    """
    Create a federated dataset with IID users from the Alpaca dataset.

    Users are created by first sampling the dataset length from
    ``user_dataset_len_sampler`` and then sampling the datapoints IID.
    """
    # Load your training data from JSON file
    train_dataset_file_path = os.path.join(script_dir, "dataset", "e1_easy_train_data.json")
    with open(train_dataset_file_path, "r") as f:
        train_data = json.load(f)
        
    hf_train_dataset = HFDataset.from_list(train_data)
    train_input_ids, train_labels = preprocess_hate_speech(hf_train_dataset, tokenizer)
    train_user_dataset = partition_data(train_input_ids, train_labels,
                                      user_dataset_len_sampler)

    train_federated_dataset = make_iid_federated_dataset(
        train_user_dataset, tokenizer, dataloader_kwargs)
    
     # Determine the number of training users
    num_train_users = len(train_user_dataset)
    num_eval_users = max(1, int(num_train_users * 0.1))
    
    # Load your evaluation data from JSON file
    eval_dataset_file_path = os.path.join(script_dir, "dataset", "e1_easy_eval_data.json")
    with open(eval_dataset_file_path, "r") as f:
        eval_data = json.load(f)

    hf_eval_dataset = HFDataset.from_list(eval_data)
    eval_input_ids, eval_labels = preprocess_hate_speech(hf_eval_dataset, tokenizer)
    
    # Partition evaluation data among num_eval_users
    eval_user_dataset = partition_data(
        eval_input_ids, eval_labels,
        user_dataset_len_sampler
    )                                           
                                            
    val_federated_dataset = make_iid_federated_dataset(eval_user_dataset,
                                                       tokenizer,
                                                       dataloader_kwargs)
    
    central_dataset = make_central_dataset(eval_user_dataset, tokenizer)
    logger.info(f"# of train users = {len(train_user_dataset)}, "
                f"# of val users = {len(eval_user_dataset)}")

    return train_federated_dataset, val_federated_dataset, central_dataset, {}
