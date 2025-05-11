import logging
import json
import os
from uuid import uuid4

import numpy as np
import torch
import transformers
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from peft import LoraConfig, PeftConfig, TaskType
from dataset.argument_parsing import add_dataset_arguments, get_datasets, parse_draw_num_datapoints_per_user, get_custom_datasets
from model.hugging_face import add_special_tokens, causal_lm_metrics_fn, smart_embedding_resize, wrap_hugging_face_model
from utils.argument_parsing import (
    add_algorithm_arguments,
    add_filepath_arguments,
    add_seed_arguments,
    get_algorithm,
    maybe_inject_arguments_from_config,
    parse_mechanism,
)
from utils.callback.hugging_face import HuggingFaceModelCheckpointingCallback
from utils.logging import init_logging
from utils.argument_parsing import parse_mechanism
from dataset.argument_parsing import get_custom_datasets  

# from llm.argument_parsing import add_llm_arguments, parse_central_lr_scheduler, parse_peft_config
from pfl.aggregate.simulate import SimulatedBackend
from pfl.callback import AggregateMetricsToDisk, CentralEvaluationCallback, StopwatchCallback, WandbCallback
from pfl.hyperparam import NNEvalHyperParams, NNTrainHyperParams
from pfl.algorithm import (
    AdaptMuOnMetricCallback,
    FederatedAlgorithm,
    FederatedAveraging,
    FedProx,
    FedProxParams,
    NNAlgorithmParams,
)
from pfl.internal.ops import pytorch_ops
from pfl.model.pytorch import PyTorchModel

import json
from datasets import Dataset as HFDataset
from typing import Dict, Any, Tuple
from pfl.data.pytorch import PyTorchDataDataset, PyTorchFederatedDataset
from pfl.data.sampling import get_user_sampler
# from hugging_face import IGNORE_INDEX, GetItemDataset, UserIDCollatorWrapper #Check
from types import SimpleNamespace


# Initialize logging, seeds, etc.
init_logging(logging.INFO)
logger = logging.getLogger(__name__)

# Define a function that runs a single training experiment.
def run_training_experiment(noise_cohort_size,round):
    
    # np.random.seed(42)
    # torch.manual_seed(42)
    central_privacy = parse_mechanism(
        mechanism_name="gaussian_moments_accountant", 
        clipping_bound=0.1,
        epsilon=2.0,
        delta=1e-6,
        order=2,
        cohort_size=100,                # your fixed cohort size
        noise_cohort_size=noise_cohort_size,  # variable for this experiment
        num_epochs=100,
        population=1e6,
        min_separation=None,
        is_central=True
    )
    
    # Define the tokenizer.
    tokenizer = transformers.AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    model_max_length=512,
    padding_side="right",
    use_fast=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = '<s>'
    
    # Load and prepare dataset.
    training_data, val_data, central_data, metadata = get_custom_datasets(tokenizer)
    
    # Load and prepare the model.
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1", 
        quantization_config=bnb_config, 
        device_map="auto", 
        trust_remote_code=True
    )
    hf_model.config.pad_token_id = tokenizer.pad_token_id
    
    # Parameter efficient fine-tuning - same parameters as centralized learning
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

    hf_model = wrap_hugging_face_model(hf_model, peft_config,
                                       causal_lm_metrics_fn)
    # Put on GPU if available.
    hf_model = hf_model.to(pytorch_ops.get_default_device())

    params = [p for p in hf_model.parameters() if p.requires_grad]
    central_optimizer = torch.optim.Adam(params,
                                         0.01, #edit here
                                         eps=1.0e-4,
                                         betas=(0.9, 0.99))


    #cosine is default for LLM
    central_learning_rate_scheduler = get_cosine_schedule_with_warmup(
                central_optimizer,
                num_warmup_steps=10,
                num_training_steps=100, #should be equal to central_num_iterations
                num_cycles=0.5) #default

    model = PyTorchModel(
        model=hf_model,
        local_optimizer_create=torch.optim.SGD,
        central_optimizer=central_optimizer,
        central_learning_rate_scheduler=central_learning_rate_scheduler,
        amp_dtype=getattr(torch, "bfloat16"),
        grad_scaling=False,
        model_dtype_same_as_amp=False,
        use_torch_compile=True)

    backend = SimulatedBackend(training_data=training_data,
                               val_data=val_data,
                               postprocessors=[central_privacy])#central_privacy
    
    # Training and evaluation hyperparameters.
    model_train_params = NNTrainHyperParams(
    local_learning_rate=0.01, #edit here
    local_num_epochs=1,
    local_batch_size=16, #control here
    local_max_grad_norm=1.0,
    grad_accumulation_steps=1)

    model_eval_params = NNEvalHyperParams(
        local_batch_size=8)

    callbacks = [
        StopwatchCallback(),
        CentralEvaluationCallback(central_data, model_eval_params=model_eval_params, frequency=1),
        AggregateMetricsToDisk(f'./metrics_noise_cohort_size_{round}/{noise_cohort_size}.csv'),
        # Set checkpoint frequency equal to 50 (i.e., only at the final epoch)
        HuggingFaceModelCheckpointingCallback(hf_model, tokenizer, f"hf_checkpoint/pfl_noise_cohort_size_new_{round}/{noise_cohort_size}", checkpoint_frequency=5)
    ]
    
    #Algorithm-Fedavg
    algorithm_params = NNAlgorithmParams(
                central_num_iterations=100, #around 20th epochs, it processes about 30,000 samples, which match the number of samples for centralized learning with 2 epoch
                evaluation_frequency=1,
                train_cohort_size=100, #you might want to change this
                val_cohort_size=0)
    algorithm = FederatedAveraging()

    
    # Run the federated averaging algorithm.
    logger.info("Starts federated learning.")
    model = algorithm.run(
        algorithm_params=algorithm_params,
        backend=backend,
        model=model,
        model_train_params=model_train_params,
        model_eval_params=model_eval_params,
        callbacks=callbacks
    )
    
    # Optionally, you could return the model or any relevant metrics.
    return model

# Now, run experiments for different noise cohort sizes.
# noise_cohort_values = [100, 500, 1000, 5000, 10000, 50000]
# epsilons = [0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
# clipping_bounds = [0.05, 0.1, 0.5, 1.0, 10.0]
for i in range(2, 3):
    noise_cohort_values = [100, 500]
    for nc in noise_cohort_values:
        logger.info(f"Running experiment with noise_cohort_size = {nc}")
        # The output directory for checkpointing can include the noise cohort value.
        final_model = run_training_experiment(noise_cohort_size=nc, round = i)
        # # You might also push the final model to Hub here, if desired:
        # final_model.push_to_hub(f"Saef/mistral_attention_dp_nc{nc}")
