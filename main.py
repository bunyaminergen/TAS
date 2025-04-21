# Standard library imports
import os

# Third-party imports
from omegaconf import OmegaConf

# Local imports
from src.utils.log.manager import Logger
from src.utils.type.schema import RootConfig
from src.utils.data.manager import DataManager
from src.utils.common.helpers import Timer, GpuMemoryReleaser
from src.process.train import SupervisedFineTuning, Distillation
from src.model.core import (TeacherFactory, Assistant, Student, BatchGenerationStrategy, SingleGenerationStrategy,
                            DDPGenerationStrategy)


def main():
    """
    Main function to run the end-to-end process of the model training and inference.
    """

    # Logger Initialization
    log_manager = Logger(name="Main")
    logger = log_manager.get()

    # Timer Initialization
    timer = Timer()
    timer.start()
    logger.info("End-to-end process started.")

    # Gpu Memory Releaser Initialization
    gpu_memory = GpuMemoryReleaser()

    # Configuration
    config_path = "src/config/config.yaml"
    config_load = OmegaConf.load(config_path)
    conf_dict = OmegaConf.to_container(config_load, resolve=True)
    config = RootConfig(**conf_dict)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = config.cuda.cuda_alloc_conf

    # Data Management
    data_manager = DataManager(config)
    data_path = data_manager.run()

    # Teacher
    teacher_factory = TeacherFactory(config, data_path, model_type="open")

    ## Teacher Data Generation
    # teacher = teacher_factory.build()
    # teacher.set_generate_strategy(DDPGenerationStrategy())
    # teacher.generate()
    # gpu_memory.release(teacher_factory)

    ## Teacher Data Download
    teacher_factory.download()

    # Assistant
    assistant = Assistant(config)

    ## Base Model Inference
    assistant.load(model="Base")
    assistant.inference("Give me a quick an example of bubble sort in Python.")

    ## SFT Training
    # sft = SupervisedFineTuning(config=config, assistant=assistant)
    # sft.train()
    # gpu_memory.release(sft)

    ## SFT Model Inference
    assistant.load(model="Assistant")
    assistant.inference("Give me a quick an example of bubble sort in Python.")

    # Student
    student = Student(config)

    ## Knowledge Distillation
    # assistant.load(model="Assistant")
    # distiller = Distillation(config=config, assistant=assistant, student=student)
    # distiller.distillate()
    # student.save()
    # student.load()

    ## Student Inference
    student.inference("Write a bubble‑sort implementation in Python.")
    gpu_memory.release(assistant)
    gpu_memory.release(student)

    # Timer Termination
    timer.end()
    hours, minutes, seconds = timer.calculate()
    logger.info(f"End-to-end process took {hours}h {minutes}m {seconds:.3f}s.")


if __name__ == "__main__":
    main()
