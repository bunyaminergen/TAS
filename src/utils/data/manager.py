# Standard library imports
import os
import json
from typing import Optional, Union, Protocol, Annotated

# Third-party imports
from datasets import load_dataset, DownloadConfig, Dataset, DatasetDict

# Local imports
from src.utils.type.schema import RootConfig
from src.utils.log.manager import Logger


class DatasetItemProtocol(Protocol):
    """
    Protocol for dataset item definitions.

    This interface defines the required structure of each dataset item,
    which includes identifying keys and optional paths.
    """

    @property
    def key(self) -> str:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def subset(self) -> Optional[str]:
        ...

    @property
    def external_local_path(self) -> Optional[str]:
        ...


class DataManager:
    """
    DataManager handles downloading, loading, saving, and extracting prompts from datasets.

    Parameters
    ----------
    config : RootConfig
        Configuration object containing dataset and prompt settings.

    Attributes
    ----------
    config : RootConfig
        The loaded configuration.
    prompt_length_threshold : int
        Maximum length of a prompt in words.
    logger : logging.Logger
        Logger instance for the class.
    raw_data_dir : str
        Directory for raw dataset files.
    arrow_dir : str
        Directory for datasets in Arrow format.
    jsonl_dir : str
        Directory for datasets in JSONL format.
    prompt_dir : str
        Directory for storing extracted prompts.
    prompts_file : str
        File path for the prompts file.
    datasets_list : list[DatasetItemProtocol]
        List of dataset configurations.
    download_config : DownloadConfig
        HuggingFace download configuration.
    """

    def __init__(self, config: RootConfig):
        self.config = config
        self.prompt_length_threshold = config.teacher.base.prompt_length
        self.logger = Logger(name="DataManager").get()
        self._skip_datasets = False

        self.raw_data_dir = os.path.join(".data", "dataset", "raw")
        os.makedirs(self.raw_data_dir, exist_ok=True)

        self.arrow_dir = os.path.join(self.raw_data_dir, "arrow")
        os.makedirs(self.arrow_dir, exist_ok=True)

        self.jsonl_dir = os.path.join(self.raw_data_dir, "jsonl")
        os.makedirs(self.jsonl_dir, exist_ok=True)

        self.prompt_dir = os.path.join(self.raw_data_dir, "prompt")
        os.makedirs(self.prompt_dir, exist_ok=True)

        self.prompts_file = os.path.join(self.prompt_dir, "prompts.jsonl")
        self.datasets_list: list[DatasetItemProtocol] = config.datasets or []

        self.download_config = DownloadConfig(
            num_proc=4,
            max_retries=5,
            resume_download=True,
        )

    def _load(
            self,
            dataset_dir_name: Annotated[str, "Directory-safe dataset name"]
    ) -> Optional[Union[Dataset, DatasetDict]]:
        """
        Load dataset from JSONL if it exists.

        Parameters
        ----------
        dataset_dir_name : str
            Directory-safe name of the dataset.

        Returns
        -------
        Optional[Union[Dataset, DatasetDict]]
            The loaded dataset or None if not found.
        """
        if not isinstance(dataset_dir_name, str):
            raise TypeError("Expected str for parameter dataset_dir_name")

        jsonl_files = [
            f for f in os.listdir(self.jsonl_dir)
            if f.startswith(dataset_dir_name) and f.endswith(".jsonl")
        ]
        if not jsonl_files:
            return None

        single_split_path = os.path.join(self.jsonl_dir, f"{dataset_dir_name}.jsonl")
        if len(jsonl_files) == 1 and jsonl_files[0] == f"{dataset_dir_name}.jsonl":
            self.logger.info("Single split loaded in JSONL format: %s", single_split_path)
            return load_dataset("json", data_files={"train": single_split_path}, split="train")

        data_files = {
            f.replace(".jsonl", "").replace(dataset_dir_name, "").lstrip("_"): os.path.join(self.jsonl_dir, f)
            for f in jsonl_files
        }
        self.logger.info("Multiple splits loaded in JSONL format: %s", list(data_files.keys()))
        return load_dataset("json", data_files=data_files)

    def _download(
            self,
            dataset_name: Annotated[str, "Name of the dataset"],
            subset: Annotated[Optional[str], "Subset name"] = None,
            jsonl: Annotated[bool, "Whether to save as JSONL"] = True,
            external_local_path: Annotated[Optional[str], "Path to local dataset"] = None
    ) -> Union[Dataset, DatasetDict]:
        """
        Download dataset or load it from disk or local path.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset.
        subset : Optional[str]
            Subset name.
        jsonl : bool
            Whether to save the dataset as JSONL format.
        external_local_path : Optional[str]
            Path to a local dataset.

        Returns
        -------
        Union[Dataset, DatasetDict]
            The downloaded or loaded dataset.
        """
        if not isinstance(dataset_name, str):
            raise TypeError("Expected str for parameter dataset_name")

        dataset_dir_name = dataset_name.replace('/', '_')
        arrow_dataset_dir = os.path.join(self.arrow_dir, dataset_dir_name)

        ds_jsonl = self._load(dataset_dir_name)
        if ds_jsonl is not None:
            self.logger.info(
                "Found existing JSONL for dataset %s. Loading from JSONL and skipping Arrow download.",
                dataset_dir_name
            )
            return ds_jsonl

        if os.path.exists(arrow_dataset_dir):
            self.logger.info("Found existing Arrow dataset at: %s", arrow_dataset_dir)
            ds = load_dataset(arrow_dataset_dir)

        elif external_local_path and os.path.exists(external_local_path):
            self.logger.info("Loading dataset from local path: %s", external_local_path)
            ds = load_dataset(external_local_path, download_config=self.download_config)

        else:
            self.logger.info("Downloading dataset '%s' from Hugging Face...", dataset_name)
            ds = (
                load_dataset(dataset_name, subset, download_config=self.download_config)
                if subset else
                load_dataset(dataset_name, download_config=self.download_config)
            )
            ds.save_to_disk(arrow_dataset_dir)
            self.logger.info("Dataset saved in Arrow format at: %s", arrow_dataset_dir)

        if jsonl:
            self._save(ds, dataset_dir_name)

        return ds

    def _save(
            self,
            ds: Annotated[Union[Dataset, DatasetDict], "Dataset to save"],
            dataset_dir_name: Annotated[str, "Directory-safe name"]
    ) -> None:
        """
        Save dataset to JSONL format.

        Parameters
        ----------
        ds : Union[Dataset, DatasetDict]
            Dataset to be saved.
        dataset_dir_name : str
            Directory-safe name for the dataset.
        """
        self.logger.info("Saving dataset '%s' to JSONL format...", dataset_dir_name)

        if not isinstance(dataset_dir_name, str):
            raise TypeError("Expected str for parameter dataset_dir_name")

        if isinstance(ds, DatasetDict):
            for split_name, split_dataset in ds.items():
                jsonl_path = os.path.join(self.jsonl_dir, f"{dataset_dir_name}_{split_name}.jsonl")
                if os.path.exists(jsonl_path):
                    self.logger.info("JSONL already exists: %s", jsonl_path)
                    continue
                self.logger.info("Saving split='%s' to %s", split_name, jsonl_path)
                split_dataset.to_json(jsonl_path, lines=True, orient="records")
        else:
            jsonl_path = os.path.join(self.jsonl_dir, f"{dataset_dir_name}.jsonl")
            if os.path.exists(jsonl_path):
                self.logger.info("JSONL already exists: %s", jsonl_path)
            else:
                self.logger.info("Saving single split to %s", jsonl_path)
                ds.to_json(jsonl_path, lines=True, orient="records")

    def _extract(
            self,
            ds: Annotated[Union[Dataset, DatasetDict], "Dataset to extract prompts from"],
            dataset_item: DatasetItemProtocol,
    ) -> None:
        """
        Extract prompt data from dataset based on key.

        Parameters
        ----------
        ds : Union[Dataset, DatasetDict]
            Dataset to extract data from.
        dataset_item : DatasetItemProtocol
            Metadata about the dataset including key for extraction.
        """
        key = dataset_item.key
        if not key:
            self.logger.warning("No 'key' specified for dataset '%s'. Skipping extraction.", dataset_item.name)
            return

        def process(entry):
            val = entry.get(key, "")
            if val and len(val.split()) <= self.prompt_length_threshold:
                return {"instruction": val}
            return None

        with open(self.prompts_file, "a", encoding="utf-8") as f:
            if isinstance(ds, DatasetDict):
                for split_name, split_dataset in ds.items():
                    self.logger.info("Extracting from '%s' split '%s'", dataset_item.name, split_name)
                    for item in split_dataset:
                        record = process(item)
                        if record:
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                self.logger.info("Extracting from single-split dataset '%s'", dataset_item.name)
                for item in ds:
                    record = process(item)
                    if record:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def run(
            self,
            jsonl: Annotated[bool, "Whether to generate JSONL if not exists"] = True
    ) -> Annotated[str, "Path to the generated prompts file"]:
        """
        Run the data processing pipeline to extract prompts.

        Parameters
        ----------
        jsonl : bool
            Whether to generate JSONL file if not exists.

        Returns
        -------
        str
            File path to the generated prompts JSONL file.

        Examples
        --------
        >>> from src.utils.type.schema import RootConfig
        >>> config = RootConfig(...)
        >>> test_dm = DataManager(config)
        >>> test_dm.run()
        '.data/dataset/raw/prompt/prompts.jsonl'
        """
        if os.path.exists(self.prompts_file):
            self.logger.info("Prompts file already exists at %s. Skipping generation.", self.prompts_file)
            return self.prompts_file

        for ds_item in self.datasets_list:
            ds_obj = self._download(
                dataset_name=ds_item.name,
                subset=ds_item.subset,
                external_local_path=ds_item.external_local_path,
                jsonl=jsonl
            )
            self._extract(ds_obj, ds_item)

        return self.prompts_file


if __name__ == "__main__":
    from omegaconf import OmegaConf

    config_path = "src/config/config.yaml"
    config_load = OmegaConf.load(config_path)
    conf_dict = OmegaConf.to_container(config_load, resolve=True)
    root_config = RootConfig(**conf_dict)

    dm = DataManager(config=root_config)
    prompts_file_path = dm.run()
    print("Prompts file path:", prompts_file_path)
