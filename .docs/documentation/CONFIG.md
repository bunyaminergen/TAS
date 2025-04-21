# Configurations

This file explains the parameters in `src/config/config.yaml` and how they are used, including their data types (
`dtype`). Below is a breakdown of each configuration section and its keys.

---

## 1. `device`

| Parameter | dtype  | Description                                                                                                | Example  |
|-----------|--------|------------------------------------------------------------------------------------------------------------|----------|
| `type`    | string | Specifies which device to use for model training and inference. Valid values are `cuda`, `cpu`, or `auto`. | `"auto"` |

---

## 2. `cuda`

| Parameter         | dtype  | Description                                                                                                | Example                      |
|-------------------|--------|------------------------------------------------------------------------------------------------------------|------------------------------|
| `cuda_alloc_conf` | string | Configuration string for CUDA memory allocation behavior (e.g. `expandable_segments:True`).                | `"expandable_segments:True"` |
| `allow_tf32`      | bool   | Whether to allow TensorFloat-32 (TF32) on Ampere GPUs for faster computation with possible precision loss. | `true`                       |

---

## 3. `teacher`

Configuration for the teacher model pipeline. Contains three sub-sections: `base`, `open`, and `closed`.

### 3.1 `teacher.base`

| Parameter       | dtype  | Description                                                              | Example                                 |
|-----------------|--------|--------------------------------------------------------------------------|-----------------------------------------|
| `prompt_length` | int    | Maximum number of tokens allowed in a prompt (soft/informational limit). | `17`                                    |
| `output_path`   | string | File path for saving teacher model outputs in JSONL format.              | `".data/dataset/teacher/teacher.jsonl"` |

### 3.2 `teacher.open`

Settings for the open-source teacher model.

#### 3.2.1 `teacher.open.model`

| Parameter             | dtype  | Description                                                       | Example                                 |
|-----------------------|--------|-------------------------------------------------------------------|-----------------------------------------|
| `name`                | string | Hugging Face model identifier for open-source teacher.            | `"stabilityai/stable-code-instruct-3b"` |
| `save_path`           | string | Local directory path where the open teacher model will be saved.  | `".model/teacher"`                      |
| `trust_remote_code`   | bool   | Whether to execute custom code from the model repository.         | `True`                                  |
| `attn_implementation` | string | Attention mechanism implementation (e.g., `"flash_attention_2"`). | `"flash_attention_2"`                   |
| `bfloat16`            | bool   | Whether to use bfloat16 precision for model weights.              | `True`                                  |

#### 3.2.2 `teacher.open.pipeline`

| Parameter        | dtype | Description                                         | Example |
|------------------|-------|-----------------------------------------------------|---------|
| `max_new_tokens` | int   | Maximum tokens to generate per inference call.      | `1024`  |
| `temperature`    | float | Sampling temperature for diversity.                 | `0.5`   |
| `top_p`          | float | Nucleus sampling threshold.                         | `0.95`  |
| `top_k`          | int   | Top-k sampling cutoff.                              | `100`   |
| `use_cache`      | bool  | Enable caching of past key/value states for speed.  | `False` |
| `do_sample`      | bool  | Use sampling (`True`) or greedy decoding (`False`). | `True`  |
| `batch_size`     | int   | Number of prompts processed at once.                | `256`   |
| `resume`         | bool  | Whether to resume generation from a checkpoint.     | `true`  |

#### 3.2.3 `teacher.open.tokenizer`

| Parameter                      | dtype  | Description                                        | Example  |
|--------------------------------|--------|----------------------------------------------------|----------|
| `additional_special_tokens`    | list   | Extra tokens to add (e.g., end-of-message tokens). | `["<     |im_end|>"]`          |
| `padding_side`                 | string | Side (`"left"`/`"right"`) to apply padding.        | `"left"` |
| `add_generation_prompt`        | bool   | Prepend a generation prompt template.              | `True`   |
| `tokenize`                     | bool   | Whether to tokenize input prompts.                 | `False`  |
| `padding`                      | bool   | Enable sequence padding.                           | `True`   |
| `truncation`                   | bool   | Truncate sequences over max length.                | `True`   |
| `skip_special_tokens`          | bool   | Remove special tokens from decoded output.         | `True`   |
| `clean_up_tokenization_spaces` | bool   | Clean extra spaces post-tokenization.              | `True`   |
| `use_fast`                     | bool   | Use fast tokenizer implementation.                 | `True`   |

#### 3.2.4 `teacher.open.data`

| Parameter       | dtype  | Description                                                         | Example |
|-----------------|--------|---------------------------------------------------------------------|---------|
| `download.size` | string | Variant size of dataset to fetch from Hugging Face (e.g., `"64k"`). | `"64k"` |

### 3.3 `teacher.closed`

Settings for the closed-source (API-based) teacher model.

#### 3.3.1 `teacher.closed.model`

| Parameter | dtype  | Description                                          | Example           |
|-----------|--------|------------------------------------------------------|-------------------|
| `name`    | string | Model name for API-based teacher (e.g., OpenAI GPT). | `"gpt-3.5-turbo"` |

#### 3.3.2 `teacher.closed.pipeline`

| Parameter     | dtype | Description                         | Example |
|---------------|-------|-------------------------------------|---------|
| `temperature` | float | Sampling temperature for API calls. | `0.7`   |
| `top_p`       | float | Nucleus sampling threshold.         | `0.9`   |
| `batch_size`  | int   | Batch size for API requests.        | `1024`  |

#### 3.3.3 `teacher.closed.data`

| Parameter       | dtype  | Description                                           | Example |
|-----------------|--------|-------------------------------------------------------|---------|
| `download.size` | string | Dataset variant size for any API-based data download. | `"64k"` |

---

## 4. `assistant`

Configuration for the assistant (student-facing) model.

### 4.1 `assistant.model`

| Parameter             | dtype  | Description                                              | Example                              |
|-----------------------|--------|----------------------------------------------------------|--------------------------------------|
| `name`                | string | Hugging Face model identifier for the assistant.         | `"Qwen/Qwen2.5-Coder-1.5B-Instruct"` |
| `save_path`           | string | Local directory path where the assistant model is saved. | `".model/assistant/base"`            |
| `attn_implementation` | string | Attention implementation method.                         | `"flash_attention_2"`                |
| `bfloat16`            | bool   | Use bfloat16 precision along with quantization.          | `true`                               |

### 4.2 `assistant.tokenizer`

| Parameter             | dtype  | Description                                  | Example  |
|-----------------------|--------|----------------------------------------------|----------|
| `padding_side`        | string | Side to pad tokens (`"left"`/`"right"`).     | `"left"` |
| `skip_special_tokens` | bool   | Remove special tokens from generated output. | `True`   |

### 4.3 `assistant.quantization`

| Parameter                   | dtype  | Description                                      | Example |
|-----------------------------|--------|--------------------------------------------------|---------|
| `load_in_4bit`              | bool   | Load model weights in 4-bit precision.           | `True`  |
| `bfloat16`                  | bool   | Use bfloat16 mixed precision with 4-bit weights. | `True`  |
| `bnb_4bit_use_double_quant` | bool   | Apply double quantization in 4-bit.              | `True`  |
| `bnb_4bit_quant_type`       | string | Quantization type (e.g., `"nf4"`, `"fp4"`).      | `"nf4"` |

### 4.4 `assistant.lora`

| Parameter      | dtype  | Description                                            | Example       |
|----------------|--------|--------------------------------------------------------|---------------|
| `lora_r`       | int    | Rank dimension (`r`) for LoRA layers.                  | `8`           |
| `lora_alpha`   | int    | Scaling factor (`alpha`) for LoRA adapters.            | `16`          |
| `lora_dropout` | float  | Dropout rate applied within LoRA layers.               | `0.05`        |
| `bias`         | string | Bias handling mode (`"none"`, `"all"`, `"lora_only"`). | `"none"`      |
| `task_type`    | string | Task type for LoRA adaptation (e.g., `"CAUSAL_LM"`).   | `"CAUSAL_LM"` |

### 4.5 `assistant.inference`

| Parameter        | dtype | Description                                  | Example |
|------------------|-------|----------------------------------------------|---------|
| `max_new_tokens` | int   | Maximum tokens to generate during inference. | `1024`  |

---

## 5. `student`

Configuration for the distilled student model.

| Parameter                 | dtype  | Description                                                                 | Example                                                     |
|---------------------------|--------|-----------------------------------------------------------------------------|-------------------------------------------------------------|
| `base_model_name`         | string | Hugging Face identifier of the base pretrained model used for distillation. | `"Qwen/Qwen2.5-Coder-1.5B-Instruct"`                        |
| `save_path`               | string | Directory where the distilled student model will be saved.                  | `".model/student"`                                          |
| `hidden_size`             | int    | Hidden layer dimension in the Transformer.                                  | `1280`                                                      |
| `intermediate_size`       | int    | Feed-forward inner layer dimension.                                         | `5120`                                                      |
| `num_hidden_layers`       | int    | Number of Transformer blocks.                                               | `24`                                                        |
| `num_attention_heads`     | int    | Number of attention heads in each block.                                    | `20`                                                        |
| `max_position_embeddings` | int    | Maximum sequence length for position embeddings.                            | `1024`                                                      |
| `repo`                    | string | Repository identifier for storing the student model code and weights.       | `"bunyaminergen/Qwen2.5-Coder-1.5B-Instruct-SFT-Distilled"` |

---

## 6. `train`

Training configurations for fine-tuning (`sft`) and knowledge distillation (`distillation`).

### 6.1 `train.sft` (fine-tuning)

| Parameter                     | dtype  | Description                                               | Example                        |
|-------------------------------|--------|-----------------------------------------------------------|--------------------------------|
| `model_name`                  | string | Alias of the model to fine-tune (e.g., `"Assistant"`).    | `"Assistant"`                  |
| `save_path`                   | string | Directory to save the fine-tuned model.                   | `".model/assistant/Assistant"` |
| `num_train_epochs`            | int    | Number of epochs for fine-tuning.                         | `11`                           |
| `per_device_train_batch_size` | int    | Training batch size per device.                           | `8`                            |
| `gradient_accumulation_steps` | int    | Steps to accumulate gradients before an optimizer update. | `2`                            |
| `learning_rate`               | float  | Initial learning rate.                                    | `1e-4`                         |
| `bfloat16`                    | bool   | Enable bfloat16 mixed precision. */                       

