import torch
import os
from datasets import Dataset
import json
import re
from tqdm import tqdm  # For progress bar
import csv
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import StoppingCriteria, StoppingCriteriaList
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from huggingface_hub import HfApi, HfFolder
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", quantization_config=bnb_config, device_map="auto")


# Load your evaluation data
with open("dataset/e1_easy_eval_data.json", "r") as f:
    eval_data = json.load(f)

# Convert to Hugging Face Dataset
hf_eval_dataset = Dataset.from_list(eval_data)

def generate_prompt(sentence):
    prompt = (f"Classify whether the sentence is hate speech, offensive, or normal, provide a list of target communities, and give explanations for classification.\n\n"
            f"### Sentence:\n{sentence}\n\n")
    return prompt


label_mapping = {0: "hate speech", 1: "normal", 2: "offensive", 3: "undecided"}
reverse_label_mapping = {v.lower(): k for k, v in label_mapping.items()}
    

def extract_output(generated_text):
    """
    Extracts classification, target communities, and explanation from the generated text.
    """
    label = None
    targets = []
    explanation = None
    
    # Split the text by sections
    parts = generated_text.split("### Classification:")
    if len(parts) > 1:
        classification_text = parts[1].strip()
        
        # Extract Classification
        sub_parts = classification_text.split("### Target Communities:")
        if len(sub_parts) > 1:
            label_section = sub_parts[0].strip()
            pattern = r"\b(hate speech|offensive|normal|undecided)\b"
            match = re.search(pattern, label_section.lower())
            if match:
                label = match.group(1)
            
            # Extract Target Communities
            target_section = sub_parts[1].split("### Explanation:")[0].strip()
            if target_section.lower() != "none":
                targets = [target.strip() for target in target_section.split(",")]

            # Extract Explanation
            explanation_section = sub_parts[1].split("### Explanation:")
            if len(explanation_section) > 1:
                explanation = explanation_section[1].strip()
    
    return label, targets, explanation


# class StopAfterExplanationCriteria(StoppingCriteria):
#     def __init__(self, tokenizer):
#         self.tokenizer = tokenizer
#         # self.prompt = prompt.lower()  # Store the lowercased prompt

#     def __call__(self, input_ids, scores, **kwargs):
#         # Decode the full generated sequence
#         generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True).lower()
        
#         # # Remove the prompt part from the generated sequence
#         # new_tokens = generated_text[len(self.prompt):].strip()
        
#         # Check if "### Explanation:" is followed by "### Sentence:"
#         explanation_end = generated_text.find("### explanation:")
#         sentence_start = generated_text.find("### sentence:", explanation_end)
        
#         # Stop if "### Sentence:" appears after "### Explanation:"
#         if explanation_end != -1 and sentence_start != -1:
#             return True
        
#         return False

class StopAfterExplanationCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        for i in range(input_ids.shape[0]):  # Loop through batch
            generated_text = self.tokenizer.decode(input_ids[i], skip_special_tokens=True).lower()

            explanation_end = generated_text.find("### explanation:")
            sentence_start = generated_text.find("### sentence:", explanation_end)

            if explanation_end != -1 and sentence_start != -1:
                return True  # One example has finished — stop the whole batch
        return False  # Continue generation



def evaluate_labels(all_predicted_labels, all_true_labels):
    # Filter out invalid predictions (where label_id == -1)
    valid_indices = [i for i, pred in enumerate(all_predicted_labels) if pred != -1]
    valid_predictions = [all_predicted_labels[i] for i in valid_indices]
    valid_true_labels = [all_true_labels[i] for i in valid_indices]

    # Compute accuracy
    accuracy = accuracy_score(valid_true_labels, valid_predictions)
    print(f"Accuracy on valid predictions: {accuracy:.2f}")

    labels = [0, 1, 2, 3]  # All possible label indices
    target_names = [label_mapping[i] for i in labels]

    # Compute detailed classification report
    output = "\nClassification Report:\n\n"
    output += classification_report(
        valid_true_labels,
        valid_predictions,
        labels=labels,
        target_names=target_names,
        zero_division=0  # Prevents division by zero warnings
    )
    return output
    

def evaluate_targets(all_predicted_targets, all_true_targets, labels):
    """
    Evaluate precision, recall, and F1 score for each sample and overall.

    Args:
        all_predicted_targets (list of list of str): Predicted labels per sample.
        all_true_labels (list of list of str): True labels per sample.

    Returns:
        dict: Contains per-sample metrics, average per-sample metrics, and overall metrics.
    """

    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    total_precision = 0
    total_recall = 0
    num_examples = len(all_true_targets)
    
    # Initialize counts for each label
    label_true_positives = {label.lower(): 0 for label in labels}
    label_false_positives = {label.lower(): 0 for label in labels}
    label_false_negatives = {label.lower(): 0 for label in labels}

    # Iterate over each sample
    for predicted, true in zip(all_predicted_targets, all_true_targets):
        # Convert labels to lower case to handle case insensitivity
        predicted_set = set(label.lower() for label in predicted)
        true_set = set(label.lower() for label in true)

        # Calculate true positives, false positives, and false negatives
        true_positives = len(true_set & predicted_set)
        false_positives = len(predicted_set - true_set)
        false_negatives = len(true_set - predicted_set)

        # Update total counts
        total_true_positives += true_positives
        total_false_positives += false_positives
        total_false_negatives += false_negatives

        # Calculate precision and recall for this example
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )

        # Accumulate precision and recall for average calculation
        total_precision += precision
        total_recall += recall
        
        for label in labels:
            label = label.lower()
            if label in true_set and label in predicted_set:
                # True Positive
                label_true_positives[label] += 1
            elif label not in true_set and label in predicted_set:
                # False Positive
                label_false_positives[label] += 1
            elif label in true_set and label not in predicted_set:
                # False Negative
                label_false_negatives[label] += 1
            # else:
            #     True Negative (not used in precision/recall calculations)


    # Calculate average precision and recall over all examples
    average_precision = total_precision / num_examples if num_examples > 0 else 0
    average_recall = total_recall / num_examples if num_examples > 0 else 0

    # Calculate overall precision and recall across all examples
    overall_precision = (
        total_true_positives / (total_true_positives + total_false_positives)
        if (total_true_positives + total_false_positives) > 0
        else 0
    )
    overall_recall = (
        total_true_positives / (total_true_positives + total_false_negatives)
        if (total_true_positives + total_false_negatives) > 0
        else 0
    )

    # Calculate F1 score from overall precision and recall
    f1_score = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) > 0
        else 0
    )
    
    # Calculate precision, recall, and F1 score for each label
    per_label_precision = {}
    per_label_recall = {}
    per_label_f1 = {}

    for label in labels:
        label = label.lower()
        tp = label_true_positives[label]
        fp = label_false_positives[label]
        fn = label_false_negatives[label]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        per_label_precision[label] = precision
        per_label_recall[label] = recall
        per_label_f1[label] = f1
    
    # Calculate macro-averaged precision, recall, and F1 score
    macro_precision = sum(per_label_precision.values()) / len(labels)
    macro_recall = sum(per_label_recall.values()) / len(labels)
    macro_f1_score = sum(per_label_f1.values()) / len(labels)
        

    return {
        'average_precision': average_precision,
        'average_recall': average_recall,
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'f1_score': f1_score,
        'per_label_precision': per_label_precision,
        'per_label_recall': per_label_recall,
        'per_label_f1': per_label_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1_score': macro_f1_score,
    }

def evaluate_explanation(all_predicted_words, all_true_words):
    """
    Compare predicted words and true words directly for evaluation. Since it does not convert to token list, this may be less robust than evaluation_expalanation_with_masks
    """
            
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    total_precision = 0
    total_recall = 0
    num_examples = len(all_true_words)
    
    for predicted_words, true_words in zip(all_predicted_words, all_true_words):
        
        # Convert labels to lower case to handle case insensitivity
        predicted_set = set(word.lower() for word in predicted_words)
        true_set = set(word.lower() for word in true_words)

        # Calculate true positives, false positives, and false negatives
        true_positives = len(true_set & predicted_set)
        false_positives = len(predicted_set - true_set)
        false_negatives = len(true_set - predicted_set)

        # Update total counts
        total_true_positives += true_positives
        total_false_positives += false_positives
        total_false_negatives += false_negatives

        # Calculate precision and recall for this example
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )

        # Accumulate precision and recall for average calculation
        total_precision += precision
        total_recall += recall
        
    # Calculate average precision and recall over all examples
    average_precision = total_precision / num_examples if num_examples > 0 else 0
    average_recall = total_recall / num_examples if num_examples > 0 else 0

    # Calculate overall precision and recall across all examples
    overall_precision = (
        total_true_positives / (total_true_positives + total_false_positives)
        if (total_true_positives + total_false_positives) > 0
        else 0
    )
    overall_recall = (
        total_true_positives / (total_true_positives + total_false_negatives)
        if (total_true_positives + total_false_negatives) > 0
        else 0
    )

    # Calculate F1 score from overall precision and recall
    f1_score = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) > 0
        else 0
    )
    
    return {
    'average_precision': average_precision,
    'average_recall': average_recall,
    'overall_precision': overall_precision,
    'overall_recall': overall_recall,
    'f1_score': f1_score,
    }

def evaluate_explanation_with_masks(all_post_tokens, all_predicted_words, all_true_words):
    """
    Convert true words and predicted words to the token list of 0 or 1 and compare their difference for evaluation.
    """
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    total_precision = 0
    total_recall = 0
    num_examples = len(all_post_tokens)

    for post_tokens, predicted_words, true_words in zip(all_post_tokens, all_predicted_words, all_true_words):
        # Convert post_tokens to lowercase for case-insensitive matching
        post_tokens_lower = [token.lower() for token in post_tokens]

        # Initialize masks with zeros
        true_mask = [0] * len(post_tokens)
        predicted_mask = [0] * len(post_tokens)

        # Helper function to mark a phrase in the mask 
        def mark_phrase_in_mask(phrase, mask):
            phrase_tokens = phrase.lower().split()
            for i in range(len(post_tokens_lower) - len(phrase_tokens) + 1):
                # Check if the slice of post_tokens matches the phrase tokens
                if post_tokens_lower[i:i + len(phrase_tokens)] == phrase_tokens:
                    for j in range(len(phrase_tokens)):
                        mask[i + j] = 1

        # Mark true words in the true mask
        for true_word in true_words:
            mark_phrase_in_mask(true_word, true_mask)

        # Mark predicted words in the predicted mask
        for pred_word in predicted_words:
            mark_phrase_in_mask(pred_word, predicted_mask)


        # Calculate true positives, false positives, and false negatives
        true_positives = sum(t and p for t, p in zip(true_mask, predicted_mask))
        false_positives = sum(p and not t for t, p in zip(true_mask, predicted_mask))
        false_negatives = sum(t and not p for t, p in zip(true_mask, predicted_mask))

        # Update total counts
        total_true_positives += true_positives
        total_false_positives += false_positives
        total_false_negatives += false_negatives

        # Calculate precision and recall for this example
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )

        # Accumulate precision and recall for average calculation
        total_precision += precision
        total_recall += recall

    # Calculate average precision and recall over all examples
    average_precision = total_precision / num_examples if num_examples > 0 else 0
    average_recall = total_recall / num_examples if num_examples > 0 else 0

    # Calculate overall precision and recall across all examples
    overall_precision = (
        total_true_positives / (total_true_positives + total_false_positives)
        if (total_true_positives + total_false_positives) > 0
        else 0
    )
    overall_recall = (
        total_true_positives / (total_true_positives + total_false_negatives)
        if (total_true_positives + total_false_negatives) > 0
        else 0
    )

    # Calculate F1 score from overall precision and recall
    f1_score = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) > 0
        else 0
    )
    # print(total_true_positives, total_false_positives, total_false_negatives)
    # print(average_precision, average_recall, overall_precision, overall_recall, f1_score)
    return {
        'average_precision': average_precision,
        'average_recall': average_recall,
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'f1_score': f1_score,
    }


def extractWords(explanation):
    """
    Extracts keywords enclosed in single quotes after 'through words'.
    """
    keywords = []
    
    # Ensure the explanation is a string
    if not isinstance(explanation, str):
        return []
    
    # Use a multiline regex to handle explanations with line breaks
    keywords_match = re.search(r"through words(.*?)$", explanation, re.DOTALL)
    if keywords_match:
        # Extract the text following "through words"
        keywords_text = keywords_match.group(1)
        # Find all words enclosed in single quotes
        keywords = re.findall(r"'(.*?)'", keywords_text)
    
    return keywords

        
#Saving the generated output
def takeCheckPoint(model_path, tokenizer, peft_model, csv_writer, output_file_path):
    labels = ["african", "arab", "asian", "caucasian", "christian", "hispanic", "buddhism", "hindu", "islam", "jewish", "men", "women", "heterosexual", "homosexual", "indigenous", "refugee", "immigrant", "disability", "none"]
    
    # Initialize lists to store predictions and true labels
    all_post_tokens = []
    all_predicted_labels = []
    all_predicted_targets=[]
    all_true_labels = []
    all_true_targets = []
    all_predicted_words = []
    all_true_words = []
    
    # Move model to device and set to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    peft_model.to(device)
    peft_model.eval()
    
    with open(output_file_path, mode="w", newline='', encoding='utf-8') as csv_file:
        
        writer = csv.writer(csv_file)
        
        writer.writerow(["Index", "Sentence", "New_Tokens", "True_Label", "Predicted_Label", "True_Targets", "Predicted_Targets", "True_Words", "Predicted_Words"])
        

        for idx, example in enumerate(tqdm(hf_eval_dataset, desc="Processing Examples")):
            # Get the sentence, true label, and true targets
            sentence_tokens = example["post_tokens"]
            sentence = " ".join(sentence_tokens)
            true_label_id = example["label"]
            true_label = label_mapping.get(true_label_id, 'Unknown')
            true_targets = example["target"]
            true_explanation = example["explanation"]
            
            all_post_tokens.append(sentence_tokens)

            # Generate the prompt
            prompt = generate_prompt(sentence)

            # Tokenize the input
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

            # Create stopping criteria
            stopping_criteria = StoppingCriteriaList([StopAfterExplanationCriteria(tokenizer, prompt)])

            # Generate the prediction
            with torch.no_grad():
                outputs = peft_model.generate(
                    **inputs,
                    max_new_tokens=100,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    stopping_criteria = stopping_criteria,
                    temperature=0.7,
                    num_beams=5,
                    early_stopping=True
                )

            # Decode the generated output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(generated_text)
            # Mask the prompt and extract only new tokens
            new_tokens = generated_text[len(prompt):].strip()
            # Remove everything starting from `### Sentence:`
            cleaned_tokens = re.sub(r"### Sentence:.*", "", new_tokens, flags=re.DOTALL)

            # Extract output from generated text
            predicted_label, predicted_targets, predicted_explanation = extract_output(cleaned_tokens)

            predicted_words = extractWords(predicted_explanation)

            true_words = extractWords(true_explanation)

            # print(predicted_label_text)
            if predicted_label:
                predicted_label_id = reverse_label_mapping.get(predicted_label.lower(), -1)
                # print(predicted_label_id)
            else:
                predicted_label_id = -1  # Assign -1 if parsing failed

            if predicted_targets == []:
                predicted_targets = ["none"]

            # Append predictions and true labels
            all_predicted_labels.append(predicted_label_id)
            all_true_labels.append(true_label_id)

            # Append predicted and true targets
            all_predicted_targets.append(predicted_targets)
            all_true_targets.append(true_targets)

            #Append predicted and true explanations
            all_predicted_words.append(predicted_words)
            all_true_words.append(true_words)
            
            writer.writerow([idx, sentence, cleaned_tokens, true_label_id, predicted_label_id,  true_targets, predicted_targets, true_words, predicted_words])

    classification_report = evaluate_labels(all_predicted_labels, all_true_labels) #classification report
    target_evaluation = evaluate_targets(all_predicted_targets, all_true_targets, labels) #label evaluation
    # words_evaluation= evaluate_explanation(all_predicted_words, all_true_words)
    words_evaluation = evaluate_explanation_with_masks(all_post_tokens, all_predicted_words, all_true_words)
    
     # Write results to CSV
    csv_writer.writerow([
        model_path,
        classification_report,
        target_evaluation["average_precision"],
        target_evaluation["average_recall"],
        target_evaluation["overall_precision"],
        target_evaluation["overall_recall"],
        target_evaluation["f1_score"],
        target_evaluation['per_label_precision'],
        target_evaluation['per_label_recall'],
        target_evaluation['per_label_f1'],
        target_evaluation["macro_precision"],
        target_evaluation["macro_recall"],
        target_evaluation["macro_f1_score"],
        words_evaluation["average_precision"],
        words_evaluation["average_recall"],
        words_evaluation["overall_precision"],
        words_evaluation["overall_recall"],
        words_evaluation["f1_score"]
    ])
    
    print(f"Results saved for checkpoint: {model_path}")

#Not printing and saving the generated output
# def takeCheckPoint_2(model_path, tokenizer, peft_model, csv_writer):
#     labels = ["african", "arab", "asian", "caucasian", "christian", "hispanic", "buddhism", "hindu", "islam", "jewish", "men", "women", "heterosexual",
#               "homosexual", "indigenous", "refugee", "immigrant", "disability", "none"]
    
#     # Initialize lists to store predictions and true labels
#     all_post_tokens = []
#     all_predicted_labels = []
#     all_predicted_targets=[]
#     all_true_labels = []
#     all_true_targets = []
#     all_predicted_words = []
#     all_true_words = []
    
#     # Move model to device and set to evaluation mode
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     peft_model.to(device)
#     peft_model.eval()

#     for idx, example in enumerate(tqdm(hf_eval_dataset, desc="Processing Examples")):
#         # Get the sentence, true label, and true targets
#         sentence_tokens = example["post_tokens"]
#         sentence = " ".join(sentence_tokens)
#         true_label_id = example["label"]
#         true_label = label_mapping.get(true_label_id, 'Unknown')
#         true_targets = example["target"]
#         true_explanation = example["explanation"]

#         all_post_tokens.append(sentence_tokens)

#         # Generate the prompt
#         prompt = generate_prompt(sentence)

#         # Tokenize the input
#         inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

#         # Create stopping criteria
#         stopping_criteria = StoppingCriteriaList([StopAfterExplanationCriteria(tokenizer, prompt)])

#         # Generate the prediction
#         with torch.no_grad():
#             outputs = peft_model.generate(
#                 **inputs,
#                 max_new_tokens=100,
#                 eos_token_id=tokenizer.eos_token_id,
#                 pad_token_id=tokenizer.pad_token_id,
#                 stopping_criteria = stopping_criteria,
#                 temperature=0.7,
#                 num_beams=5,
#                 early_stopping=True
#             )

#         # Decode the generated output
#         generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         # Mask the prompt and extract only new tokens
#         new_tokens = generated_text[len(prompt):].strip()
#         # Remove everything starting from `### Sentence:`
#         cleaned_tokens = re.sub(r"### Sentence:.*", "", new_tokens, flags=re.DOTALL)

#         # Extract output from generated text
#         predicted_label, predicted_targets, predicted_explanation = extract_output(cleaned_tokens)

#         predicted_words = extractWords(predicted_explanation)

#         true_words = extractWords(true_explanation)

#         # print(predicted_label_text)
#         if predicted_label:
#             predicted_label_id = reverse_label_mapping.get(predicted_label.lower(), -1)
#             # print(predicted_label_id)
#         else:
#             predicted_label_id = -1  # Assign -1 if parsing failed

#         if predicted_targets == []:
#             predicted_targets = ["none"]

#         # Append predictions and true labels
#         all_predicted_labels.append(predicted_label_id)
#         all_true_labels.append(true_label_id)

#         # Append predicted and true targets
#         all_predicted_targets.append(predicted_targets)
#         all_true_targets.append(true_targets)

#         #Append predicted and true explanations
#         all_predicted_words.append(predicted_words)
#         all_true_words.append(true_words)

#     classification_report = evaluate_labels(all_predicted_labels, all_true_labels) #classification report
#     target_evaluation = evaluate_targets(all_predicted_targets, all_true_targets, labels) #label evaluation
#     # words_evaluation= evaluate_explanation(all_predicted_words, all_true_words)
#     words_evaluation = evaluate_explanation_with_masks(all_post_tokens, all_predicted_words, all_true_words)
    
#      # Write results to CSV
#     csv_writer.writerow([
#         model_path,
#         classification_report,
#         target_evaluation["average_precision"],
#         target_evaluation["average_recall"],
#         target_evaluation["overall_precision"],
#         target_evaluation["overall_recall"],
#         target_evaluation["f1_score"],
#         target_evaluation['per_label_precision'],
#         target_evaluation['per_label_recall'],
#         target_evaluation['per_label_f1'],
#         target_evaluation["macro_precision"],
#         target_evaluation["macro_recall"],
#         target_evaluation["macro_f1_score"],
#         words_evaluation["average_precision"],
#         words_evaluation["average_recall"],
#         words_evaluation["overall_precision"],
#         words_evaluation["overall_recall"],
#         words_evaluation["f1_score"]
#     ])
    
#     print(f"Results saved for checkpoint: {model_path}")
    
def takeCheckPoint_2(model_path, tokenizer, peft_model, csv_writer):
    labels = ["african", "arab", "asian", "caucasian", "christian", "hispanic", "buddhism", "hindu", "islam", "jewish", "men", "women", "heterosexual",
              "homosexual", "indigenous", "refugee", "immigrant", "disability", "none"]
    
    # Initialize lists to store predictions and true labels
    all_post_tokens = []
    all_predicted_labels = []
    all_predicted_targets=[]
    all_true_labels = []
    all_true_targets = []
    all_predicted_words = []
    all_true_words = []
    
    # Move model to device and set to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    peft_model.to(device)
    peft_model.eval()
    
    batch_size = 32
    batched_inputs = []
    batched_prompts = []
    batched_examples = []

    for idx, example in enumerate(tqdm(hf_eval_dataset, desc="Processing Examples")):
        # Get the sentence, true label, and true targets
        sentence_tokens = example["post_tokens"]
        sentence = " ".join(sentence_tokens)
        true_label_id = example["label"]
        true_label = label_mapping.get(true_label_id, 'Unknown')
        true_targets = example["target"]
        true_explanation = example["explanation"]

        # Generate the prompt
        prompt = generate_prompt(sentence)
        stopping_criteria = StoppingCriteriaList([StopAfterExplanationCriteria(tokenizer)])

        # Tokenize the input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        inputs["input_ids"] = inputs["input_ids"].squeeze(0)
        inputs["attention_mask"] = inputs["attention_mask"].squeeze(0)
        
        #New
        batched_inputs.append((inputs["input_ids"], inputs["attention_mask"]))
        batched_prompts.append(prompt)
        batched_examples.append((example, sentence_tokens, true_label_id, true_targets, true_explanation))

        if len(batched_inputs) == batch_size or idx == len(hf_eval_dataset) - 1:
            input_ids = [i for i, _ in batched_inputs]
            attention_masks = [m for _, m in batched_inputs]

            batch = tokenizer.pad(
                {"input_ids": input_ids, "attention_mask": attention_masks},
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = peft_model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_new_tokens=100,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    num_beams=5,
                    temperature=0.7,
                    early_stopping=True,
                    stopping_criteria = stopping_criteria
                )
            
            for output, prompt, (example, sentence_tokens, true_label_id, true_targets, true_explanation) in zip(outputs, batched_prompts, batched_examples):
                generated_text = tokenizer.decode(output, skip_special_tokens=True)
                new_tokens = generated_text[len(prompt):].strip()
                cleaned_tokens = re.sub(r"### Sentence:.*", "", new_tokens, flags=re.DOTALL)

                predicted_label, predicted_targets, predicted_explanation = extract_output(cleaned_tokens)
                predicted_words = extractWords(predicted_explanation)
                true_words = extractWords(true_explanation)

                if predicted_label:
                    predicted_label_id = reverse_label_mapping.get(predicted_label.lower(), -1)
                else:
                    predicted_label_id = -1

                if predicted_targets == []:
                    predicted_targets = ["none"]

                # Append everything
                all_post_tokens.append(sentence_tokens)
                all_predicted_labels.append(predicted_label_id)
                all_true_labels.append(true_label_id)
                all_predicted_targets.append(predicted_targets)
                all_true_targets.append(true_targets)
                all_predicted_words.append(predicted_words)
                all_true_words.append(true_words)

            # Clear batch
            batched_inputs = []
            batched_prompts = []
            batched_examples = []

    classification_report = evaluate_labels(all_predicted_labels, all_true_labels) #classification report
    target_evaluation = evaluate_targets(all_predicted_targets, all_true_targets, labels) #label evaluation
    # words_evaluation= evaluate_explanation(all_predicted_words, all_true_words)
    words_evaluation = evaluate_explanation_with_masks(all_post_tokens, all_predicted_words, all_true_words)
    
     # Write results to CSV
    csv_writer.writerow([
        model_path,
        classification_report,
        target_evaluation["average_precision"],
        target_evaluation["average_recall"],
        target_evaluation["overall_precision"],
        target_evaluation["overall_recall"],
        target_evaluation["f1_score"],
        target_evaluation['per_label_precision'],
        target_evaluation['per_label_recall'],
        target_evaluation['per_label_f1'],
        target_evaluation["macro_precision"],
        target_evaluation["macro_recall"],
        target_evaluation["macro_f1_score"],
        words_evaluation["average_precision"],
        words_evaluation["average_recall"],
        words_evaluation["overall_precision"],
        words_evaluation["overall_recall"],
        words_evaluation["f1_score"]
    ])
    
    print(f"Results saved for checkpoint: {model_path}")
    
def iterateCheckPoints(output_csv="pfl_new_evaluation_6.csv"):
    checkpoints_dir ="pfl-research/benchmarks/hf_checkpoint/final_pfl_new/pfl_4"
    # checkpoints_dir ="pfl-research/benchmarks/hf_checkpoint/pfl_noise_cohort_size_new_2/10000"
    # checkpoints_dir = "checkpoints/mistral_dp_new"
    # inference_output_path = "mistral_attention_new_inference.csv"
    
    # Open a CSV file for writing
    with open(output_csv, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # Write the header
        csv_writer.writerow([
            "Checkpoint Path",
            "Classification Report",
            "Average Precision",
            "Average Recall",
            "Overall Precision",
            "Overall Recall",
            "F1 Score",
            "Per Label Precision",
            "Per Label Recall",
            "Per Label F1",
            "Macro Precision",
            "Macro Recall",
            "Macro F1 Score",
            "Word Average Precision",
            "Word Average Recall",
            "Word Overall Precision",
            "Word Overall Recall",
            "Word F1 Score"
        ])

        
    # Iterate through each checkpoint
        
        i = 0
        for checkpoint in sorted(os.listdir(checkpoints_dir)):
            if checkpoint.startswith("."):
                continue
            #edit here
            # if "lora" not in checkpoint:
            #     continue
            checkpoint_path = os.path.join(checkpoints_dir, checkpoint)
            
            if os.path.isdir(checkpoint_path):
                print(f"Loading checkpoint from {checkpoint_path}...")

                # Load PEFT model with the checkpoint
                tokenizer = AutoTokenizer.from_pretrained(checkpoint_path) #Edit here
                tokenizer.padding_side = "left"
                # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1") #edit here
                # if tokenizer.pad_token is None:
                #     tokenizer.pad_token = tokenizer.eos_token
                peft_model = PeftModel.from_pretrained(base_model, checkpoint_path)
                
                # if i == 15:
                #     takeCheckPoint(checkpoint_path, tokenizer, peft_model, csv_writer, inference_output_path)
                # else:
                takeCheckPoint_2(checkpoint_path, tokenizer, peft_model, csv_writer)
                i +=1


def iterateCheckPoints_2(output_csv="pfl_evaluation_pfl_new_1.csv"): #ddit
    checkpoints_dir = "pfl-research/benchmarks/hf_checkpoint/final_pfl_new/" #edit
    # checkpoints_dir = "checkpoints/mistral_dp_100" #edit
    
    # Open a CSV file for writing
    with open(output_csv, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # Write the header
        csv_writer.writerow([
            "Checkpoint Path",
            "Classification Report",
            "Average Precision",
            "Average Recall",
            "Overall Precision",
            "Overall Recall",
            "F1 Score",
            "Per Label Precision",
            "Per Label Recall",
            "Per Label F1",
            "Macro Precision",
            "Macro Recall",
            "Macro F1 Score",
            "Word Average Precision",
            "Word Average Recall",
            "Word Overall Precision",
            "Word Overall Recall",
            "Word F1 Score"
        ])

        
    # Iterate through each checkpoint
        for pfl in sorted(os.listdir(checkpoints_dir)):
            if pfl.startswith("."):
                continue
      
            pfl_path = os.path.join(checkpoints_dir, pfl)
        
            for checkpoint in os.listdir(pfl_path):
                if checkpoint.startswith("."):
                    continue
                    
                checkpoint_path = os.path.join(pfl_path, checkpoint)
                if "99" not in checkpoint_path:
                    continue
            
                if os.path.isdir(checkpoint_path):
                    print(f"Loading checkpoint from {checkpoint_path}...")

                    # Load PEFT model with the checkpoint
                    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path) #Edit here
                    tokenizer.padding_side = "left"
                    # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1") #edit here
                    peft_model = PeftModel.from_pretrained(base_model, checkpoint_path)
                    takeCheckPoint_2(checkpoint_path, tokenizer, peft_model, csv_writer)

def iterateCheckPointsSingle(output_csv="single.csv"):
    # checkpoints_dir = "pfl-research/benchmarks/hf_checkpoint/final_fl_new/5"
    checkpoint_path = "checkpoints/mistral_attention_new/checkpoint-24100"
    # inference_output_path = "mistral_attention_new_inference.csv"
    
    # Open a CSV file for writing
    with open(output_csv, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # Write the header
        csv_writer.writerow([
            "Checkpoint Path",
            "Classification Report",
            "Average Precision",
            "Average Recall",
            "Overall Precision",
            "Overall Recall",
            "F1 Score",
            "Per Label Precision",
            "Per Label Recall",
            "Per Label F1",
            "Macro Precision",
            "Macro Recall",
            "Macro F1 Score",
            "Word Average Precision",
            "Word Average Recall",
            "Word Overall Precision",
            "Word Overall Recall",
            "Word F1 Score"
        ])

        
    # Iterate through each checkpoint
        
      
            
        if os.path.isdir(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}...")

            # Load PEFT model with the checkpoint
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path) #Edit here
            tokenizer.padding_side = "left"
            # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1") #edit here
            # if tokenizer.pad_token is None:
            #     tokenizer.pad_token = tokenizer.eos_token
            peft_model = PeftModel.from_pretrained(base_model, checkpoint_path)

            # if i == 15:
            #     takeCheckPoint(checkpoint_path, tokenizer, peft_model, csv_writer, inference_output_path)
            # else:
            takeCheckPoint_2(checkpoint_path, tokenizer, peft_model, csv_writer)

                
iterateCheckPoints()