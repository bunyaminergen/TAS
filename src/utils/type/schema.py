# Standard library imports
import logging
from typing import List, Optional, Literal, Dict, Union

# Third-party imports
from pydantic import BaseModel, StrictBool, Field, conint, confloat, constr


class CudaConfig(BaseModel):
    cuda_alloc_conf: constr(min_length=1) = Field("expandable_segments:True", description="Configuration for CUDA memor"
                                                                                          "y allocation.")
    allow_tf32: StrictBool = Field(True, description="Whether to allow TF32 for faster matmul on supported GPUs.")


class DeviceConfig(BaseModel):
    type: Literal["cuda", "cpu", "auto"] = Field("cuda", description="Device type to use.")


class TeacherBaseConfig(BaseModel):
    prompt_length: conint(gt=0) = Field(29, description="Maximum number of words/tokens in a prompt (for base usage).")
    output_path: constr(min_length=1) = Field(..., description="Output path for teacher-generated JSONL file.")


class TeacherModelConfig(BaseModel):
    name: constr(min_length=1) = Field(..., description="Hugging Face model name/path for the teacher.")
    save_path: constr(min_length=1) = Field(..., description="Local path where the teacher model is/will be saved.")
    trust_remote_code: StrictBool = Field(True, description="Whether to allow custom codefrom the model's repo.")
    attn_implementation: Optional[str] = Field("flash_attention_2", description="Attention implementation method.")
    bfloat16: StrictBool = Field(True, description="Whether to use bfloat16 precision for the teacher model.")


class TeacherPipelineConfig(BaseModel):
    max_new_tokens: conint(ge=1) = Field(512, description="Number of new tokens to generate during inference.")
    temperature: confloat(ge=0.0, le=100.0) = Field(0.3, description="Sampling temperature; higher produces more varied"
                                                                     " outputs.")
    top_p: confloat(ge=0.0, le=1.0) = Field(0.95, description="Nucleus sampling 'top_p' value.")
    top_k: conint(ge=0) = Field(50, description="Top-k sampling cutoff.")
    use_cache: StrictBool = Field(True, description="Whether to use cache during generation (for faster inference).")
    do_sample: StrictBool = Field(True, description="Whether to sample the output (if False, uses greedy decoding).")
    batch_size: conint(ge=1) = Field(1, description="Number of instructions to process simultaneously in a batch.")
    resume: StrictBool = Field(True, description="Whether to resume from the latest data checkpoint if exists.")


class TeacherTokenizerConfig(BaseModel):
    additional_special_tokens: List[str] = Field(["<|im_end|>"], description="Additional special tokens to add to the t"
                                                                             "okenizer.")
    padding_side: Literal["left", "right"] = Field("left", description="Which side to pad on (left or right).")
    add_generation_prompt: StrictBool = Field(True, description="Whether to add generation prompt for chat template.")
    tokenize: StrictBool = Field(False, description="If True, apply tokenization for the input prompt.")
    padding: StrictBool = Field(True, description="If True, apply padding.")
    truncation: StrictBool = Field(True, description="If True, apply truncation.")
    skip_special_tokens: StrictBool = Field(True, description="If True, skip special tokens while decoding.")
    clean_up_tokenization_spaces: StrictBool = Field(True, description="If True, cleanup tokenization spaces while deco"
                                                                       "ding.")
    use_fast: StrictBool = Field(True, description="If True, use fast tokenizer implementation.")


class TeacherDataDownloadConfig(BaseModel):
    size: Literal["16k", "32k", "64k"] = Field("64k", description="Which size of the dataset to download from HF.")


class TeacherDataConfig(BaseModel):
    download: TeacherDataDownloadConfig = Field(..., description="Download config for teacher dataset.")


class TeacherOpenConfig(BaseModel):
    model: TeacherModelConfig
    pipeline: TeacherPipelineConfig
    tokenizer: TeacherTokenizerConfig
    data: TeacherDataConfig


class TeacherClosedModelConfig(BaseModel):
    name: constr(min_length=1) = Field(..., description="OpenAI model name (e.g., 'gpt-3.5-turbo').")


class TeacherClosedPipelineConfig(BaseModel):
    temperature: confloat(ge=0.0, le=100.0) = Field(0.7)
    top_p: confloat(ge=0.0, le=1.0) = Field(0.9)
    batch_size: conint(ge=1) = Field(2)


class TeacherClosedDataConfig(BaseModel):
    download: TeacherDataDownloadConfig


class TeacherClosedConfig(BaseModel):
    model: TeacherClosedModelConfig
    pipeline: TeacherClosedPipelineConfig
    data: TeacherClosedDataConfig


class TeacherFactoryParams(BaseModel):
    model_type: Literal["open", "closed"] = Field("open", description="Model Type: open or closed.")


class TeacherConfig(BaseModel):
    base: TeacherBaseConfig
    open: TeacherOpenConfig
    closed: TeacherClosedConfig


class AssistantModelConfig(BaseModel):
    name: constr(min_length=1) = Field(..., description="Hugging Face model name/path for the assistant.")
    save_path: constr(min_length=1) = Field(..., description="Local path where the assistant model will be saved.")
    attn_implementation: Optional[str] = Field("flash_attention_2", description="Attention implementation method.")
    bfloat16: StrictBool = Field(True, description="Whether to use bfloat16 precision for the assistant model.")


class AssistantTokenizerConfig(BaseModel):
    padding_side: Literal["left", "right"] = Field("left", description="Which side to pad on (left or right).")
    skip_special_tokens: StrictBool = Field(True, description="Whether to skip special tokens while decoding.")


class AssistantQuantizationConfig(BaseModel):
    load_in_4bit: StrictBool = Field(True, description="Whether to load the assistant model in 4-bit precision.")
    bfloat16: StrictBool = Field(True, description="Whether to use bfloat16 precision for 4-bit quantization.")
    bnb_4bit_use_double_quant: StrictBool = Field(True, description="Double quantization for 4-bit.")
    bnb_4bit_quant_type: constr(min_length=1) = Field("nf4", description="Quantization data type used for 4-bit.")


class AssistantLoraConfig(BaseModel):
    lora_r: conint(gt=0) = Field(8, description="Rank dimension (r) for LoRA.")
    lora_alpha: conint(gt=0) = Field(16, description="Alpha (scaling factor) for LoRA.")
    lora_dropout: confloat(ge=0.0, le=1.0) = Field(0.05, description="LoRA dropout rate; must be between 0.0 and 1.0.")
    bias: Literal["none", "all", "lora_only"] = Field("none", description="LoRA bias type.")
    task_type: constr(min_length=1) = Field("CAUSAL_LM", description="Task type for LoRA (e.g., 'CAUSAL_LM').")


class AssistantInferenceConfig(BaseModel):
    max_new_tokens: conint(gt=0) = Field(1024, description="Number of new tokens to generate for assistant inference.")


class AssistantConfig(BaseModel):
    model: AssistantModelConfig
    tokenizer: AssistantTokenizerConfig
    quantization: AssistantQuantizationConfig
    lora: AssistantLoraConfig
    inference: AssistantInferenceConfig


class StudentConfig(BaseModel):
    base_model_name: constr(min_length=1) = Field(..., description="HF model name/path for the student model.")
    save_path: constr(min_length=1) = Field(..., description="Local path where the student model will be saved.")
    hidden_size: conint(gt=0) = Field(..., description="Hidden size of the student model.")
    intermediate_size: conint(gt=0) = Field(..., description="Intermediate size of the student model.")
    num_hidden_layers: conint(gt=0) = Field(..., description="Number of hidden layers in the student model.")
    num_attention_heads: conint(gt=0) = Field(..., description=" Number of attention heads in the student model.")
    max_position_embeddings: conint(gt=0) = Field(1024, description="Maximum position embeddings for student model.")
    repo: constr(min_length=1) = Field(..., description="Hugging Face repo name for the student model.")


class TrainExperimentConfig(BaseModel):
    name: constr(min_length=1) = Field(..., description="MLflow experiment name.")
    path: constr(min_length=1) = Field(".logs/mlruns", description="Path for MLflow logging.")
    run_name: constr(min_length=1) = Field(..., description="Run name for MLflow.")
    set_tracking_uri: constr(min_length=1) = Field("file:.logs/mlruns", description="Tracking URI for MLflow.")
    dataset_artifact_path: constr(min_length=1) = Field("dataset", description="Artifact path for dataset logs.")
    model_artifact_path: constr(min_length=1) = Field("huggingface_model", description="Artifact path for model logs.")
    registered_model_name: constr(min_length=1) = Field("QwenLoRA", description="Name for the registered model in MLflo"
                                                                                "w.")
    dataset_tags: Dict[str, str] = Field({}, description="Extra tags for dataset logging.")


class TrainDatasetConfig(BaseModel):
    seed: conint(gt=0) = Field(19, description="Random seed for dataset splitting.")
    test_size: confloat(gt=0.0, lt=1.0) = Field(0.1, description="Proportion for the test dataset.")
    train_batched: StrictBool = Field(True, description="Whether to batch the training dataset.")
    eval_batched: StrictBool = Field(True, description="Whether to batch the evaluation dataset.")


class TrainTokenizerConfig(BaseModel):
    truncation: StrictBool = Field(True, description="Whether to truncate the sequences.")
    max_length: conint(gt=0) = Field(1024, description="Maximum sequence length for tokenization.")
    mlm: StrictBool = Field(False, description="Whether to use masked language modeling.")


class TrainPEFTLoraConfig(BaseModel):
    r: conint(gt=0) = Field(16, description="Lora rank (dimensionality reduction factor).")
    lora_alpha: conint(ge=0) = Field(32, description="Lora scaling factor.")
    lora_dropout: confloat(ge=0, le=1) = Field(0.05, description="Dropout probability for Lora layers.")
    target_modules: Union[str, List[str]] = Field("all-linear", description="Target modules to apply LoRA.")
    modules_to_save: List[str] = Field(["lm_head", "embed_token"], description="Modules to exclude from LoRA and keep i"
                                                                               "n original form.")
    task_type: str = Field("CAUSAL_LM", description="The task type for PEFT. For example, 'CAUSAL_LM'.")
    bias: Literal["none", "all", "lora_only"] = Field("none", description="LoRA bias type.")


class TrainPEFTConfig(BaseModel):
    lora: TrainPEFTLoraConfig


class TrainSFTConfig(BaseModel):
    model_name: constr(min_length=1) = Field(..., description="Name of the model to fine-tune.")
    save_path: constr(min_length=1) = Field(..., description="Directory to save the fine-tuned model.")
    num_train_epochs: conint(gt=0) = Field(1, description="Number of training epochs.")
    per_device_train_batch_size: conint(gt=0) = Field(1, description="Batch size per GPU/TPU core.")
    gradient_accumulation_steps: conint(gt=0) = Field(1, description="Number of updates steps to accumulate before perf"
                                                                     "orming a backward pass.")
    learning_rate: float = Field(1e-4, description="Learning rate for the optimizer.")
    bfloat16: StrictBool = Field(True, description="Whether to use bfloat16 for training.")
    fp16: StrictBool = Field(True, description="Whether to use fp16 for training.")
    eval_strategy: constr(min_length=1) = Field("epoch", description="Evaluation strategy (e.g. 'epoch').")
    report_to: constr(min_length=1) = Field("mlflow", description="Where to report training logs (e.g. 'mlflow').")
    overwrite_output_dir: StrictBool = Field(True, description="If True, overwrites the output directory.")
    logging_steps: conint(gt=0) = Field(10, description="Log every X updates steps.")
    save_steps: conint(gt=0) = Field(50, description="Save a checkpoint every X updates steps.")
    experiment: TrainExperimentConfig
    dataset: TrainDatasetConfig
    tokenizer: TrainTokenizerConfig
    peft: TrainPEFTConfig


class TrainKnowledgeDistillationLossConfig(BaseModel):
    temperature: confloat(ge=0.0, le=1.0) = Field(1.0, description="Temperature for KD loss.")
    reduction: Literal["batchmean", "sum", "mean", "none"] = Field("batchmean", description="Reduction type for KD loss"
                                                                                            ".")


class TrainKnowledgeDistillationConfig(BaseModel):
    weight: confloat(ge=0.0, le=1.0) = Field(0.5, description="Weight for the KD loss.")
    loss: TrainKnowledgeDistillationLossConfig


class TrainDistillationConfig(BaseModel):
    output_dir: constr(min_length=1) = Field(".model/student/StudentDistilled", description="Directory for distillation"
                                                                                            " output.")
    overwrite_output_dir: StrictBool = Field(True, description="If True, overwrites the distillation output directory.")
    num_train_epochs: conint(gt=0) = Field(1, description="Number of training epochs.")
    per_device_train_batch_size: conint(gt=0) = Field(2, description="Batch size per device.")
    gradient_accumulation_steps: conint(gt=0) = Field(4, description="Number of steps to accumulate gradients.")
    learning_rate: float = Field(1e-4, description="Learning rate during distillation.")
    bf16: StrictBool = Field(True, description="Whether to use bf16 for distillation.")
    evaluation_strategy: constr(min_length=1) = Field("epoch", description="When to run evaluation (e.g. 'epoch').")
    logging_steps: conint(gt=0) = Field(10, description="Log every X updates steps.")
    save_steps: conint(gt=0) = Field(50, description="Save every X updates steps.")
    report_to: constr(min_length=1) = Field("mlflow", description="Reporting integration (e.g. 'mlflow').")
    run_name: constr(min_length=1) = Field("KnowledgeDistillation", description="Name of the distillation run.")
    knowledgedistillation: TrainKnowledgeDistillationConfig
    experiment: TrainExperimentConfig


class TrainConfig(BaseModel):
    sft: TrainSFTConfig
    distillation: TrainDistillationConfig


class DatasetItem(BaseModel):
    name: constr(min_length=1) = Field(..., description="Hugging Face dataset name (e.g., 'username/dataset_name').")
    key: constr(min_length=1) = Field(..., description="Key inside the dataset from which to extract prompts.")
    subset: Optional[constr(min_length=1)] = Field("default", description="Optional dataset subset or config name.")
    external_local_path: Optional[constr(min_length=1)] = Field(None, description="Local path to dataset, if provided.")


class RootConfig(BaseModel):
    cuda: CudaConfig = Field(default_factory=CudaConfig, description="CUDA-related configuration settings.")
    device: DeviceConfig = Field(default_factory=DeviceConfig, description="Specifies the device for computations.")
    teacher: TeacherConfig
    assistant: AssistantConfig
    student: StudentConfig
    datasets: List[DatasetItem]
    train: TrainConfig


class LoggerType(BaseModel):
    name: constr(min_length=1) = Field(..., description="Name of the logger.")
    app: constr(min_length=1) = Field("Ayvaz", description="Name of the application.")
    path: constr(min_length=1) = Field(".logs", description="Folder where log files are stored.")
    file: constr(min_length=1) = Field(".log", description="Name of the log file.")
    console_level: conint(ge=0) = Field(logging.DEBUG, description="Console log level.")
    file_level: conint(ge=0) = Field(logging.DEBUG, description="File log level.")
    max_bytes: conint(gt=0) = Field(5000000, description="Maximum log file size (bytes) before rotation.")
    backup_count: conint(gt=0) = Field(5, description="Number of old log files to keep.")
    verbose: StrictBool = Field(True, description="Whether to output logs to the console.")


class SingleLineConsoleFormatterType(BaseModel):
    app: constr(min_length=1) = Field("Ayvaz", description="Name of the application.")
    date_format: Optional[str] = Field(None, description="Date format for console logs.")


class SingleLineFileFormatterType(BaseModel):
    app: constr(min_length=1) = Field("Ayvaz", description="Name of the application.")
    date_format: Optional[str] = Field(None, description="Date format for file logs.")


class PromptDetail(BaseModel):
    system: constr(min_length=20, max_length=1024) = Field(..., description=("System prompt text guiding the model to p"
                                                                             "roduce docstrings. Must be between 20 and"
                                                                             " 1024 characters."))


class PromptConfig(BaseModel):
    Teacher: PromptDetail = Field(..., description="Prompt config for the teacher, containing a single 'system' message"
                                                   ".")
    Assistant: PromptDetail = Field(..., description="Prompt config for the assistant, containing a single 'system' mes"
                                                     "sage.")
