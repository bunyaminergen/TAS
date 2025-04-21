# Standard library imports
from typing import Annotated, Union, Optional

# Third-party imports
import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments

# Local imports
from src.utils.log.manager import Logger
from src.utils.type.schema import TrainKnowledgeDistillationConfig


class KnowledgeDistillation(Trainer):
    """
    Custom HuggingFace Trainer that implements Knowledge Distillation.

    This trainer enhances the student model training by incorporating
    soft targets from an assistant (teacher) model using KL Divergence.

    Parameters
    ----------
    assistant : torch.nn.Module
        The assistant model used for distillation.
    config : TrainKnowledgeDistillationConfig
        Configuration containing temperature, reduction method, and weight.
    model : torch.nn.Module
        The student model being trained.
    args : TrainingArguments
        HuggingFace TrainingArguments.
    train_dataset : Dataset, optional
        Dataset used for training.
    eval_dataset : Dataset, optional
        Dataset used for evaluation.
    processing_class : optional
        Optional processor class.
    data_collator : optional
        Optional collator function for batching data.
    **kwargs : dict
        Additional arguments for HuggingFace Trainer.

    Attributes
    ----------
    logger : logging.Logger
        Internal logger for tracking training behavior.
    distillation_weight : float
        Weight factor balancing CE and KD loss.
    temperature : float
        Temperature value used to soften logits.
    reduction : str
        Reduction strategy for KL divergence loss.

    Examples
    --------
    >>> assistant = DummyModel(num_labels=4)
    >>> test_student = DummyModel(num_labels=4)
    >>> args = TrainingArguments(output_dir='./temp')
    >>> test_cfg = TrainKnowledgeDistillationConfig(...)
    >>> trainer = KnowledgeDistillation(assistant, cfg, student, args)
    """

    def __init__(
            self,
            assistant: Annotated[torch.nn.Module, "Teacher model"],
            config: Annotated[TrainKnowledgeDistillationConfig, "KD config"],
            model: Annotated[torch.nn.Module, "Student model"],
            args: Annotated[TrainingArguments, "TrainingArguments"],
            train_dataset: Optional[torch.utils.data.Dataset] = None,
            eval_dataset: Optional[torch.utils.data.Dataset] = None,
            processing_class=None,
            data_collator=None,
            **kwargs,
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            data_collator=data_collator,
            **kwargs
        )

        log_manager = Logger(name="KnowledgeDistillationLoss")
        self.logger = log_manager.get()
        self.logger.info("Knowledge Distillation initialized.")

        self.config = config
        self.assistant = assistant
        self.distillation_weight = config.weight
        self.temperature = config.loss.temperature
        self.reduction = config.loss.reduction

        for p in self.assistant.parameters():
            p.requires_grad = False
        self.assistant.eval()
        self.logger.info("Assistant model parameters are frozen.")

    def kl_divergence_loss(
            self,
            student_logits: Annotated[torch.Tensor, "Logits from the student model"],
            assistant_logits: Annotated[torch.Tensor, "Logits from the assistant model"]
    ) -> Annotated[torch.Tensor, "KL divergence loss value"]:
        """
        Compute the KL divergence loss between student and assistant logits.

        Parameters
        ----------
        student_logits : torch.Tensor
            Logits produced by the student model.
        assistant_logits : torch.Tensor
            Logits produced by the assistant model.

        Returns
        -------
        torch.Tensor
            Computed KL divergence loss.

        Examples
        --------
        >>> s = torch.randn(3, 5)
        >>> a = torch.randn(3, 5)
        >>> test_cfg = TrainKnowledgeDistillationConfig(...)
        >>> kd = KnowledgeDistillation(test_assistant, cfg, student)
        >>> loss = kd.kl_divergence_loss(s, a)
        >>> isinstance(loss, torch.Tensor)
        True
        """
        st_logits_temp = student_logits / self.temperature
        as_logits_temp = assistant_logits / self.temperature

        student_probs = F.log_softmax(st_logits_temp, dim=-1)
        assistant_probs = F.softmax(as_logits_temp, dim=-1)

        kl_div_loss = F.kl_div(
            student_probs,
            assistant_probs,
            reduction=self.reduction
        ) * (self.temperature ** 2)

        self.logger.debug(
            f"KL Divergence shapes; Student: {student_probs.shape}, "
            f"Assistant: {assistant_probs.shape}, "
            f"Loss: {kl_div_loss.item():.3f}"
        )

        return kl_div_loss

    def compute_loss(
            self,
            model: Annotated[torch.nn.Module, "Student model"],
            inputs: Annotated[dict, "Batch input data"],
            return_outputs: Annotated[bool, "Flag to return model output"] = False,
            **kwargs
    ) -> Annotated[Union[torch.Tensor, tuple], "Total loss or (loss, model output)"]:
        """
        Compute combined CE and KD loss for the student model.

        Parameters
        ----------
        model : torch.nn.Module
            The student model.
        inputs : dict
            Input batch containing 'input_ids' and 'labels'.
        return_outputs : bool, optional
            Whether to return model outputs along with loss.

        Returns
        -------
        torch.Tensor or (torch.Tensor, Any)
            The loss value or a tuple of (loss, outputs).
        """
        out_s = model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"]
        )
        ce = out_s.loss

        with torch.no_grad():
            out_a = self.assistant(
                input_ids=inputs["input_ids"],
            )

        kd = self.kl_divergence_loss(out_s.logits, out_a.logits)
        loss = (1 - self.distillation_weight) * ce + self.distillation_weight * kd

        return (loss, out_s) if return_outputs else loss


if __name__ == "__main__":
    from src.utils.type.schema import TrainKnowledgeDistillationLossConfig

    loss_cfg = TrainKnowledgeDistillationLossConfig(
        temperature=0.8,
        reduction="batchmean"
    )
    cfg = TrainKnowledgeDistillationConfig(
        weight=0.4,
        loss=loss_cfg
    )


    class DummyModel(torch.nn.Module):
        """
        Dummy model for testing purposes.
        """

        def __init__(self, num_labels: int):
            super().__init__()
            self.num_labels = num_labels

        def forward(self, input_ids, labels=None):
            """
            Dummy forward method that simulates model output.
            """
            batch_size = input_ids.size(0)
            logits = torch.randn(batch_size, self.num_labels)
            loss = torch.tensor(0.8) if labels is not None else None
            return type("O", (), {"logits": logits, "loss": loss})


    test_num_labels = 4
    student = DummyModel(test_num_labels)
    test_assistant = DummyModel(test_num_labels)

    train_args = TrainingArguments(
        output_dir=".temp",
        per_device_train_batch_size=2,
        num_train_epochs=1,
        logging_steps=1,
    )

    kd_trainer = KnowledgeDistillation(
        assistant=test_assistant,
        config=cfg,
        model=student,
        args=train_args,
    )

    s_logits = torch.randn(3, test_num_labels)
    a_logits = torch.randn(3, test_num_labels)
    print("KL Divergence Loss:", kd_trainer.kl_divergence_loss(s_logits, a_logits).item())

    batch = {
        "input_ids": torch.randint(0, 100, (4, 10)),
        "labels": torch.randint(0, test_num_labels, (4,))
    }
    print("Total (KD + CE) Loss:", kd_trainer.compute_loss(student, batch).item())
