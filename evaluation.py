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

#Load the model
checkpoint_path = "pfl-research/benchmarks/hf_checkpoint/pfl_4_retry_3/checkpoint-49"
base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", quantization_config=bnb_config, device_map="auto")
# base_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path) #Edit here
                # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1") #edit here
peft_model = PeftModel.from_pretrained(base_model, checkpoint_path)
            
# Load your evaluation data
with open("dataset/e1_easy_eval_data.json", "r") as f:
    eval_data = json.load(f)
eval_data = eval_data[:100]

# Convert to Hugging Face Dataset
hf_eval_dataset = Dataset.from_list(eval_data)

def generate_prompt(sentence):
    prompt = (f"Classify whether the sentence is hate speech, offensive, or normal.\n\n"
            f"### Sentence:\n{sentence}\n\n")
    return prompt


label_mapping = {0: "hate speech", 1: "normal", 2: "offensive", 3: "undecided"}
reverse_label_mapping = {v.lower(): k for k, v in label_mapping.items()}


def extract_targets(sentence):
    sentence = sentence.strip()

    # Step 1: Remove '###' at the end if present
    sentence = re.sub(r'###$', '', sentence)

    # Step 2: Remove special characters except commas
    sentence = re.sub(r'[^A-Za-z0-9, ]+', '', sentence)

    # Step 3: Remove extra whitespace
    sentence = ' '.join(sentence.split())
    
    targets = sentence.split(",")
    if len(targets)  < 1:
        return []
    cleaned_targets = [target.strip() for target in targets]
    return cleaned_targets
    
def extract_output(generated_text):
    
    """
    Extracts the classification label from the generated text.
    Assumes the label is one of the predefined classes.
    """
    label = None
    targets = []
    # Define a regex pattern to capture the label
    parts = generated_text.split("### Classification:")
    if len(parts) > 1:
        classification_text = parts[1].strip()
        sub_parts = classification_text.split("### Target Communities:")
        if len(sub_parts) > 1:
            targets = extract_targets(sub_parts[1])
        pattern = r"\b(hate speech|offensive|normal|undecided)\b"
        match = re.search(pattern, sub_parts[0].lower())
        if match:
            label = match.group(1)
    return label, targets  


# Define custom stopping criterion that halts when "###" appears again after "Target communities"
class StopOnHashCriteriaAfterTarget(StoppingCriteria):
    def __init__(self, stop_token, after_token, tokenizer):
        self.stop_token = stop_token.lower()
        self.after_token = after_token.lower()
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Decode generated token IDs back to a readable string
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True).lower()
        
        # Stop if "###" appears after "Target communities"
        if self.after_token in generated_text:
            return self.stop_token in generated_text.split(self.after_token)[-1]  # Only consider "###" after "Target communities"
        return False


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

def takeCheckPoint(model_name, tokenizer, peft_model, csv_writer, output_file_path):
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
    

    with open(output_file_path, mode="a", newline='', encoding='utf-8') as csv_file: #changed from w to a in append mode
        
        writer = csv.writer(csv_file)
        
        writer.writerow(["Index", "Sentence", "New_Tokens", "True_Label", "Predicted_Label", "True_Targets", "Predicted_Targets"])
    
        for idx, example in enumerate(tqdm(hf_eval_dataset, desc="Processing Examples")):
            # Get the sentence, true label, and true targets
            sentence_tokens = example["post_tokens"]
            sentence = " ".join(sentence_tokens)
            true_label_id = example["label"]
            true_label = label_mapping.get(true_label_id, 'Unknown')
            true_targets = example["target"]
            
            all_post_tokens.append(sentence_tokens)

            # Generate the prompt
            prompt = generate_prompt(sentence)

            # Tokenize the input
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            
            # Create the stopping criterion
            stopping_criteria = StoppingCriteriaList([StopOnHashCriteriaAfterTarget("###", "Target Communities:", tokenizer)])

            # Generate the prediction
            with torch.no_grad():
                outputs = peft_model.generate(
                    **inputs,
                    max_new_tokens=150,
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
            predicted_label, predicted_targets = extract_output(cleaned_tokens)
            
            
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
            
            writer.writerow([idx, sentence, cleaned_tokens, true_label_id, predicted_label_id,  true_targets, predicted_targets])
            csv_file.flush()
    
    classification_report = evaluate_labels(all_predicted_labels, all_true_labels) #classification report
    target_evaluation = evaluate_targets(all_predicted_targets, all_true_targets, labels) #label evaluation
    
     # Write results to CSV
    csv_writer.writerow([
        model_name,
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
    ])
    

def iterateCheckPoints(model_name = "pfl Mistral", output_csv="simple_evaluation.csv"):
    # checkpoints_dir = "pfl-research/benchmarks/hf_checkpoint/fl_2"
    inference_output_path = "pfl_retry_3.csv"
    # Open a CSV file for writing
    with open(output_csv, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # Write the header
        csv_writer.writerow([
            "Model Name",
            "Classification Report",
            "Target Average Precision",
            "Target Average Recall",
            "Target Overall Precision",
            "Target Overall Recall",
            "Target F1 Score",
            "Target Per Label Precision",
             "Target Per Label Recall",
            "Target Per Label F1",
            "Target Macro Precision",
            "Target Macro Recall",
            "Target Macro F1 Score",
        ])

        takeCheckPoint(model_name, tokenizer, peft_model, csv_writer, inference_output_path)
            

iterateCheckPoints()