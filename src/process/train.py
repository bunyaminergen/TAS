# Standard library imports
import os
from typing import Annotated, Tuple

# Third-party imports
import mlflow
import pandas as pd
import mlflow.transformers
from peft import LoraConfig
from datasets import Dataset
from omegaconf import OmegaConf
from datasets import load_dataset
from pydantic import ValidationError
from trl import SFTConfig, SFTTrainer
from mlflow.data.pandas_dataset import from_pandas
from transformers import TrainingArguments, DataCollatorForLanguageModeling

# Local imports
from src.utils.log.manager import Logger
from src.model.core import Assistant, Student
from src.model.loss import KnowledgeDistillation
from src.utils.type.schema import RootConfig, PromptConfig


class SupervisedFineTuning:
    """
    Conducts supervised fine-tuning (SFT) on a language model using PEFT with LoRA.

    This class loads datasets, formats and tokenizes them, sets up training with
    a PEFT configuration, and performs model training.

    Parameters
    ----------
    config : RootConfig
        Root configuration object containing all fine-tuning parameters.
    assistant : Assistant
        Assistant object containing model and tokenizer.

    Attributes
    ----------
    logger : logging.Logger
        Logger for logging fine-tuning steps.
    assistant : Assistant
        The language model assistant instance.
    config : RootConfig
        Configuration settings.
    """

    def __init__(
            self,
            config: Annotated[RootConfig, "Root configuration"],
            assistant: Annotated[Assistant, "Assistant model and tokenizer"],
    ):
        log_manager = Logger(name="SupervisedFineTuningAssistant")
        self.logger = log_manager.get()

        self.config = config
        self.assistant = assistant

        # SFT configs
        fine_tuning_conf = config.train.sft
        self.data_path = config.teacher.base.output_path
        self.save_path = fine_tuning_conf.save_path
        self.overwrite_output_dir = fine_tuning_conf.overwrite_output_dir
        self.num_train_epochs = fine_tuning_conf.num_train_epochs
        self.per_device_train_batch_size = fine_tuning_conf.per_device_train_batch_size
        self.gradient_accumulation_steps = fine_tuning_conf.gradient_accumulation_steps
        self.learning_rate = fine_tuning_conf.learning_rate
        self.bfloat16 = fine_tuning_conf.bfloat16
        self.fp16 = fine_tuning_conf.fp16
        self.eval_strategy = fine_tuning_conf.eval_strategy
        self.logging_steps = fine_tuning_conf.logging_steps
        self.save_steps = fine_tuning_conf.save_steps
        self.report_to = fine_tuning_conf.report_to
        self.experiment_conf = fine_tuning_conf.experiment
        self.dataset_conf = fine_tuning_conf.dataset
        self.tokenizer_conf = fine_tuning_conf.tokenizer

        # LoRA config
        peft_lora = fine_tuning_conf.peft.lora
        self.r = peft_lora.r
        self.lora_alpha = peft_lora.lora_alpha
        self.lora_dropout = peft_lora.lora_dropout
        self.target_modules = peft_lora.target_modules
        self.modules_to_save = peft_lora.modules_to_save
        self.task_type = peft_lora.task_type

        # Mlflow config
        os.makedirs(self.experiment_conf.path, exist_ok=True)
        mlflow.set_tracking_uri(self.experiment_conf.set_tracking_uri)
        mlflow.set_experiment(self.experiment_conf.name)

        self.logger.info("FineTuning instance initialized.")

    def _load_dataset(self) -> Annotated[Tuple[Dataset, Dataset], "Train and test datasets"]:
        """
        Load and split the dataset into training and test sets.

        Returns
        -------
        (Dataset, Dataset)
            Tuple containing train and test datasets.
        """
        self.logger.info("Loading dataset from %s", self.config.teacher.base.output_path)
        dataset_dict = load_dataset("json", data_files=self.data_path)["train"]
        self.logger.info("Splitting dataset into train and evaluation sets")
        split_dataset = dataset_dict.train_test_split(
            test_size=self.dataset_conf.test_size,
            seed=self.dataset_conf.seed
        )
        return split_dataset["train"], split_dataset["test"]

    @staticmethod
    def _format_dataset(
            example: Annotated[dict, "A single dataset row"]
    ) -> Annotated[dict, "Formatted input with prompt"]:
        """
        Format the dataset row using prompt template.

        Parameters
        ----------
        example : dict
            Dictionary containing 'instruction' and 'output' keys.

        Returns
        -------
        dict
            A dictionary with a single 'text' key for formatted prompt.
        """
        try:
            cfg = OmegaConf.load("src/config/prompt.yaml")
            raw_data = OmegaConf.to_container(cfg, resolve=True)
            validated_prompts = PromptConfig(**raw_data)
            assistant_system_prompt = validated_prompts.Assistant.system
        except (FileNotFoundError, ValidationError) as e:
            raise ValueError(f"Problem with prompt.yaml config: {e}")

        user_str = example["instruction"]
        assistant_str = example["output"]
        return {"text": f"System: {assistant_system_prompt}\nUser: {user_str}\nAssistant: {assistant_str}"}

    def _tokenize(
            self,
            examples: Annotated[dict, "Dictionary with 'text' field"]
    ) -> Annotated[dict, "Tokenized output"]:
        """
        Tokenize examples using the assistant's tokenizer.

        Parameters
        ----------
        examples : dict
            Dictionary with 'text' field.

        Returns
        -------
        dict
            Tokenized output from tokenizer.
        """
        return self.assistant.tokenizer(
            examples["text"],
            truncation=self.tokenizer_conf.truncation,
            max_length=self.tokenizer_conf.max_length
        )

    def _preprocess_dataset(
            self,
            dataset: Annotated[Dataset, "Dataset to preprocess"],
            batched: Annotated[bool, "Whether to batch examples"]
    ) -> Annotated[Dataset, "Preprocessed dataset"]:
        """
        Apply formatting and tokenization to the dataset.

        Parameters
        ----------
        dataset : Dataset
            HuggingFace dataset object.
        batched : bool
            Whether to apply preprocessing in batches.

        Returns
        -------
        Dataset
            Preprocessed dataset.
        """
        dataset = dataset.map(self._format_dataset)
        self.logger.info("Preprocessing %s dataset", dataset)

        remove_cols = [c for c in dataset.column_names if c != "text"]
        dataset = dataset.map(
            self._tokenize,
            batched=batched,
            remove_columns=remove_cols
        )
        self.logger.info("%s dataset preprocessed", dataset)

        return dataset

    def _add_trainer(
            self,
            train_dataset: Annotated[Dataset, "Preprocessed training set"],
            eval_dataset: Annotated[Dataset, "Preprocessed evaluation set"]
    ) -> Annotated[SFTTrainer, "SFTTrainer instance for training"]:
        """
        Initialize the trainer with model, datasets, and configuration.

        Parameters
        ----------
        train_dataset : Dataset
            Training data.
        eval_dataset : Dataset
            Evaluation data.

        Returns
        -------
        SFTTrainer
            Trainer for fine-tuning the model.
        """
        self.logger.info("Initializing Trainer")

        peft_config = LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            modules_to_save=self.modules_to_save,
            task_type=self.task_type,
        )

        training_args = SFTConfig(
            output_dir=self.save_path,
            overwrite_output_dir=self.overwrite_output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            bf16=self.bfloat16,
            fp16=self.fp16,
            eval_strategy=self.eval_strategy,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            report_to=self.report_to,
            run_name=self.experiment_conf.run_name,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.assistant.tokenizer,
            mlm=self.tokenizer_conf.mlm
        )

        return SFTTrainer(
            model=self.assistant.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            peft_config=peft_config,
        )

    def train(self) -> None:
        """
        Train the model using supervised fine-tuning.

        This method orchestrates loading data, preprocessing, initializing the
        trainer, running training, and saving the resulting model and tokenizer.

        Returns
        -------
        None

        Examples
        --------
        >>> from src.model.core import Assistant
        >>> from src.utils.type.schema import RootConfig
        >>> test_config_path = "src/config/config.yaml"
        >>> test_config_load = OmegaConf.load(config_path)
        >>> test_conf_dict = OmegaConf.to_container(config_load, resolve=True)
        >>> config = RootConfig(**conf_dict)
        >>> assistant = Assistant(config)
        >>> sft = SupervisedFineTuning(test_conf_dict, assistant)
        >>> sft.train()
        """
        self.logger.info("Starting training process")
        train_dataset, eval_dataset = self._load_dataset()

        train_dataset = self._preprocess_dataset(train_dataset, batched=self.dataset_conf.train_batched)
        eval_dataset = self._preprocess_dataset(eval_dataset, batched=self.dataset_conf.eval_batched)

        trainer = self._add_trainer(train_dataset, eval_dataset)

        with mlflow.start_run(run_name=self.experiment_conf.run_name):
            mlflow.log_artifact(
                local_path=self.data_path,
                artifact_path=self.experiment_conf.dataset_artifact_path
            )

            df = pd.read_json(self.data_path, lines=True)
            dataset = from_pandas(
                df,
                source=self.data_path,
                name=self.experiment_conf.dataset_artifact_path
            )
            mlflow.log_input(dataset, context="training", tags=self.experiment_conf.dataset_tags)

            self.logger.info("Training started")
            trainer.train()
            self.logger.info("Training completed")

            trainer.model.save_pretrained(self.save_path)
            self.assistant.tokenizer.save_pretrained(self.save_path)


class Distillation:
    """
    Implements knowledge distillation from a teacher (assistant) model to a student model.

    Loads a dataset, formats and tokenizes it, and distills knowledge using
    a custom `KnowledgeDistillation` trainer.

    Parameters
    ----------
    config : RootConfig
        Root configuration object for the distillation process.
    assistant : Assistant
        The teacher model with tokenizer.
    student : Student
        The student model with tokenizer.

    Attributes
    ----------
    logger : logging.Logger
        Logger instance for tracking distillation steps.
    data_path : str
        Path to the training data.
    output_dir : str
        Directory where the distilled model and tokenizer will be saved.
    """

    def __init__(
            self,
            config: Annotated[RootConfig, "Root configuration object"],
            assistant: Annotated[Assistant, "Teacher model"],
            student: Annotated[Student, "Student model"]
    ):
        log_manager = Logger(name="Distillation")
        self.logger = log_manager.get()

        if not isinstance(config, RootConfig):
            raise TypeError("Expected RootConfig for parameter config")
        if not isinstance(assistant, Assistant):
            raise TypeError("Expected Assistant for parameter assistant")
        if not isinstance(student, Student):
            raise TypeError("Expected Student for parameter student")

        self.config = config
        self.assistant = assistant
        self.student = student

        self.assistant.model.eval()

        self.student_model = student.model
        self.student_tokenizer = student.tokenizer

        self.data_path = config.teacher.base.output_path
        self.logger.info(f"Distillation dataset path: {self.data_path}")

        distill_conf = config.train.distillation
        self.output_dir = distill_conf.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.logger.info("Distillation instance initialized.")

    def load_dataset(self) -> Annotated[Tuple[Dataset, Dataset], "Train and test datasets"]:
        """
        Load and split the dataset into training and evaluation sets.

        Returns
        -------
        (Dataset, Dataset)
            Tuple of training and test datasets.

        Raises
        ------
        ValueError
            If the prompt configuration file is not found or invalid.
        """
        ds = load_dataset("json", data_files=self.data_path)["train"]
        self.logger.info(f"Dataset loaded. Size: {len(ds)}")

        def format_example(example: dict) -> dict:
            """
            Format a single example into a prompt-style text input.

            This function loads a system prompt from `prompt.yaml` and combines it with the
            user instruction and assistant output from the dataset to create a full conversational
            prompt. The result is used for supervised fine-tuning or distillation tasks.

            Parameters
            ----------
            example : dict
                A dictionary containing keys 'instruction' and 'output' from the dataset.

            Returns
            -------
            dict
                A dictionary with a single key 'text', containing the formatted conversation string.

            Raises
            ------
            ValueError
                If `prompt.yaml` is missing or its structure is invalid.

            Examples
            --------
            >>> sample = {"instruction": "Tell me a joke.", "output": "Why did the chicken cross the road?"}
            >>> format_example(example)
            {'text': 'System: <system_prompt_here>\\nUser: Tell me a joke.\\nAssistant: Why did chicken cross road?'}
            """
            try:
                cfg = OmegaConf.load("src/config/prompt.yaml")
                raw_data = OmegaConf.to_container(cfg, resolve=True)
                validated_prompts = PromptConfig(**raw_data)
                assistant_system_prompt = validated_prompts.Assistant.system
            except (FileNotFoundError, ValidationError) as e:
                raise ValueError(f"Problem with prompt.yaml config: {e}")

            return {
                "text": f"System: {assistant_system_prompt}\n"
                        f"User: {example['instruction']}\n"
                        f"Assistant: {example['output']}"
            }

        ds = ds.map(format_example)
        ds_split = ds.train_test_split(test_size=0.1, seed=42)
        return ds_split["train"], ds_split["test"]

    def tokenize_fn(
            self,
            examples: Annotated[dict, "Dictionary of formatted text"]
    ) -> Annotated[dict, "Tokenized output"]:
        """
        Tokenize the text using the student tokenizer.

        Parameters
        ----------
        examples : dict
            Dictionary containing 'text' field.

        Returns
        -------
        dict
            Dictionary of tokenized output.
        """
        return self.student_tokenizer(examples["text"], truncation=True, max_length=512)

    def distillate(self) -> None:
        """
        Run the knowledge distillation process.

        Loads data, tokenizes it, initializes a knowledge distillation trainer,
        and logs artifacts with MLflow.

        Returns
        -------
        None

        Examples
        --------
        >>> from src.model.core import Assistant, Student
        >>> from src.utils.type.schema import RootConfig
        >>> test_config_path = "src/config/config.yaml"
        >>> test_config_load = OmegaConf.load(config_path)
        >>> test_conf_dict = OmegaConf.to_container(config_load, resolve=True)
        >>> config = RootConfig(**conf_dict)
        >>> assistant = Assistant(config)
        >>> student = Student(config)
        >>> distiller = Distillation(config, assistant, student)
        >>> distiller.distillate()
        """
        train_ds, eval_ds = self.load_dataset()
        train_ds = train_ds.map(
            self.tokenize_fn,
            batched=True,
            remove_columns=["text", "instruction", "output"]
        )
        eval_ds = eval_ds.map(
            self.tokenize_fn,
            batched=True,
            remove_columns=["text", "instruction", "output"]
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.student_tokenizer,
            mlm=False,
        )

        distill_args_cfg = self.config.train.distillation
        training_args = TrainingArguments(
            output_dir=distill_args_cfg.output_dir,
            overwrite_output_dir=distill_args_cfg.overwrite_output_dir,
            num_train_epochs=distill_args_cfg.num_train_epochs,
            per_device_train_batch_size=distill_args_cfg.per_device_train_batch_size,
            gradient_accumulation_steps=distill_args_cfg.gradient_accumulation_steps,
            learning_rate=distill_args_cfg.learning_rate,
            bf16=distill_args_cfg.bf16,
            evaluation_strategy=distill_args_cfg.evaluation_strategy,
            logging_steps=distill_args_cfg.logging_steps,
            save_steps=distill_args_cfg.save_steps,
            report_to=distill_args_cfg.report_to,
            run_name=distill_args_cfg.run_name,
        )

        kd_cfg = distill_args_cfg.knowledgedistillation
        kd_trainer = KnowledgeDistillation(
            assistant=self.assistant.model,
            model=self.student_model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            config=kd_cfg,
        )

        with mlflow.start_run(run_name=distill_args_cfg.run_name):
            df = pd.read_json(self.data_path, lines=True)
            dataset = from_pandas(df, source=self.data_path, name="teacher_dataset")
            mlflow.log_input(dataset, context="training")

            self.logger.info("Distillation training started ...")
            kd_trainer.train()
            self.logger.info("Distillation training finished.")

            kd_trainer.save_model(self.output_dir)
            self.student_tokenizer.save_pretrained(self.output_dir)
            self.logger.info("Student model saved → %s", self.output_dir)


if __name__ == "__main__":
    config_path = "src/config/config.yaml"
    config_load = OmegaConf.load(config_path)
    conf_dict = OmegaConf.to_container(config_load, resolve=True)
    config_test = RootConfig(**conf_dict)

    assistant_base = Assistant(config_test)
    assistant_base.load()

    fine_tuner = SupervisedFineTuning(config=config_test, assistant=assistant_base)
    fine_tuner.train()

    output_text = fine_tuner.assistant.inference(prompt="Write a simple Python class.")
    print(output_text)
