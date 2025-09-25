import json
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union, Any, Generator

import transformers
from classconfig import ConfigurableFactory, Config, ConfigurableSubclassFactory, ConfigurableMixin, ConfigurableValue, \
    RelativePathTransformer
from classconfig.classes import ConfigurableDataclassMixin
from classconfig.transforms import TryTransforms
from classconfig.validators import BoolValidator, ValueInIntervalIntegerValidator, \
    ValueInIntervalFloatValidator, AnyValidator, StringValidator, IsNoneValidator
from datasets import DatasetDict
from mlworkflows.etl.dataset import DatasetFactory
from mlworkflows.etl.tokenizer import TokenizerFactory
from mlworkflows.eval.compute_metrics import ComputeMetrics
from mlworkflows.models.hf_base import HFModelFactory
from mlworkflows.training_utils.optimizer_factory import OptimizerFactory, LionFactory
from mlworkflows.utils.modules import load_module
from transformers import AutoTokenizer, Trainer, DefaultDataCollator, \
    TrainingArguments as HFTrainingArguments, EarlyStoppingCallback, TrainerCallback
from transformers.trainer_utils import PredictionOutput

transformers.logging.set_verbosity_info()


class Workflow(ABC):
    """
    Workflows are here to define how a model should be used on given data for our task.
    It e.g. defines interface for training, evaluating, and using a model (in test time).
    """

    @abstractmethod
    def __call__(self):
        """
        Runs the workflow.
        """
        ...

    @staticmethod
    def load(path_to_workflow: str, path_to_config: Optional[str]) -> "Workflow":
        """
        Loads given workflow.

        :param path_to_workflow: Path to the workflow source file.
        :param path_to_config: Path to the configuration file.
        """

        module = load_module(path_to_workflow)

        if path_to_config is not None:
            config = Config(module.RUN, path_to=path_to_config)
        else:
            config = Config(module.RUN)

        factory = ConfigurableFactory(module.RUN)
        return factory.create({} if config is None else config.load())


class EarlyStopping(EarlyStoppingCallback):
    early_stopping_patience: int = ConfigurableValue(
        desc="Use with `metric_for_best_model` to stop training when the specified metric worsens for `early_stopping_patience` evaluation calls.",
        user_default=3,
        validator=ValueInIntervalIntegerValidator(min_value=0)
    )
    early_stopping_threshold: Optional[float] = ConfigurableValue(
        desc="Use with TrainingArguments `metric_for_best_model` and `early_stopping_patience` to denote how much the specified metric must improve to satisfy early stopping conditions.",
        user_default=0.0,
        validator=ValueInIntervalFloatValidator(min_value=0.0)
    )


class TrainingArguments(HFTrainingArguments, ConfigurableDataclassMixin):
    DESC_METADATA = "help"
    output_dir = ConfigurableValue(
        desc="The output directory where the model predictions and checkpoints will be written.",
        user_default="./training",
        transform=RelativePathTransformer(eval_counters=True)
    )
    do_train = ConfigurableValue(
        desc="Whether to run training.",
        user_default=True,
        validator=BoolValidator()
    )
    do_eval = ConfigurableValue(
        desc="Whether to run eval on the dev set.",
        user_default=True,
        validator=BoolValidator()
    )
    logging_dir: str = ConfigurableValue(
        desc="Tensorboard log dir.",
        user_default=None,
        transform=RelativePathTransformer(allow_none=True, eval_counters=True)
    )
    seed: int = ConfigurableValue(
        desc="Random seed that will be set at the beginning of training.",
        user_default=42,
        transform=lambda x: random.randint(0, 2**31 - 1) if x is None else x,
    )


class RuntimeFlopsLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs is not None:
            logs["total_flos"] = state.total_flos


class BaseTrainingWorkflow(ConfigurableMixin):
    """
    Base training workflow.
    """

    dataset_fct: DatasetFactory = ConfigurableSubclassFactory(DatasetFactory,
                                                              desc="Hugging Face dataset for training and validation.",
                                                              name="dataset",
                                                              user_default=DatasetFactory)
    model_fct: HFModelFactory = ConfigurableSubclassFactory(HFModelFactory,
                                                            desc="Hugging Face model configuration.",
                                                            name="model",
                                                            user_default=HFModelFactory)
    tokenizer_fct: Optional[TokenizerFactory] = ConfigurableSubclassFactory(TokenizerFactory,
                                                                            desc="Hugging Face tokenizer for the model. Leave empty if you wish to initialize it from the model.",
                                                                            name="tokenizer",
                                                                            voluntary=True)
    training: TrainingArguments = ConfigurableFactory(TrainingArguments, desc="Training configuration")

    compute_metrics: Optional[ComputeMetrics] = ConfigurableSubclassFactory(ComputeMetrics,
                                                                            desc="Compute metrics for evaluation.",
                                                                            voluntary=True,
                                                                            user_default=ComputeMetrics)
    resume_from_checkpoint: Optional[Union[str, bool]] = ConfigurableValue(
        "If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.",
        user_default=None,
        validator=AnyValidator([IsNoneValidator(), BoolValidator(), StringValidator()]),
        transform=TryTransforms([RelativePathTransformer(force_relative_prefix=True, allow_none=True)])
    )

    early_stopping: Optional[EarlyStopping] = ConfigurableSubclassFactory(EarlyStopping,
                                                                          desc="Early stopping configuration.",
                                                                          voluntary=True)
    optimizer_factory: OptimizerFactory = ConfigurableSubclassFactory(OptimizerFactory,
                                                                      desc="Optimizer factory. It allows to setup the optimizer. It has priority over the other optimizer arguments.",
                                                                      user_default=LionFactory)
    save_dataset: Optional[str] = ConfigurableValue(
        desc="Path to save the transformed dataset to given folder as jsonl files. If None the dataset will not be saved.",
        user_default=None,
        transform=RelativePathTransformer(allow_none=True),
        voluntary=True
    )
    save_model: Optional[str] = ConfigurableValue(
        desc="Path to save the model so it can be easily loaded with from_pretrained method. "
             "If you want to save the best model use `load_best_model_at_end`. "
             "If None the model will not be saved.",
        user_default=None,
        transform=RelativePathTransformer(allow_none=True, eval_counters=True),
        voluntary=True
    )

    def __call__(self):
        self.trainer = self._init_trainer()
        if self.save_dataset is not None:
            self.save_dataset_to_folder()

        self.trainer.train(
            resume_from_checkpoint=self.resume_from_checkpoint
        )

        if self.save_model is not None:
            self.trainer.save_model(self.save_model)

    def save_dataset_to_folder(self):
        for split in self.dataset:
            self.dataset[split].to_json(str(Path(self.save_dataset) / f"{split}.jsonl"))

    def _init_trainer(self):
        self._init_components()

        add_args = {}

        if self.training.do_train:
            if isinstance(self.dataset, DatasetDict):
                add_args["train_dataset"] = self.dataset[self.dataset_fct.train_split]
            else:
                # by default, we use the whole dataset for training
                add_args["train_dataset"] = self.dataset

        if self.training.do_eval:
            if not isinstance(self.dataset, DatasetDict):
                raise ValueError("Unable to evaluate on the dataset as there is no validation split.")
            add_args["eval_dataset"] = self.dataset[self.dataset_fct.validation_split]

            if self.compute_metrics is not None:
                add_args["compute_metrics"] = self.compute_metrics

        if self.early_stopping is not None:
            add_args["callbacks"] = [self.early_stopping]

        if self.optimizer_factory is not None:
            add_args["optimizer_cls_and_kwargs"] = (
                self.optimizer_factory.optimizer_cls,
                self.optimizer_factory.optimizer_kwargs
            )

        trainer = Trainer(
            model=self.model,
            args=self.training,
            data_collator=self.collator,
            **add_args
        )
        # this is rather dirty, but it seems like there is no other way to log flops
        trainer.callback_handler.callbacks = [RuntimeFlopsLoggingCallback()] + trainer.callback_handler.callbacks
        return trainer

    def _init_components(self):
        self._init_dataset()
        self._init_model()
        self._init_tokenizer()
        self._init_collator()

    def _init_dataset(self):
        self.dataset = self.dataset_fct.create()

    def _init_model(self):
        self.model = self.model_fct.create()

    def _init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_fct.model_path, use_fast=True,
                                                       cache_dir=self.model_fct.cache_dir) \
            if self.tokenizer_fct is None else self.tokenizer_fct.create()

    def _init_collator(self):
        self.collator = DefaultDataCollator()


class BasePredictWorkflow(ConfigurableMixin, ABC):
    """
    Base predict workflow.
    """

    dataset_fct: DatasetFactory = ConfigurableSubclassFactory(DatasetFactory,
                                                              desc="Hugging Face dataset for training and validation.",
                                                              name="dataset",
                                                              user_default=DatasetFactory)
    use_split: str = ConfigurableValue(desc="Split to use for prediction.",
                                       user_default="test",
                                       validator=StringValidator())
    model_fct: HFModelFactory = ConfigurableSubclassFactory(HFModelFactory,
                                                            desc="Hugging Face model configuration.",
                                                            name="model",
                                                            user_default=HFModelFactory)
    tokenizer_fct: Optional[TokenizerFactory] = ConfigurableSubclassFactory(TokenizerFactory,
                                                                            desc="Hugging Face tokenizer for the model. Leave empty if you wish to initialize it from the model.",
                                                                            name="tokenizer",
                                                                            voluntary=True)

    per_device_batch_size: int = ConfigurableValue("Batch size per GPU/TPU/MPS/NPU core/CPU for evaluation.",
                                                   user_default=8,
                                                   validator=ValueInIntervalIntegerValidator(min_value=1))

    predictions: Optional[str] = ConfigurableValue(
        desc="Path to directory for results",
        transform=RelativePathTransformer(allow_none=True),
        voluntary=True
    )

    def __call__(self):
        self.trainer = self._init_trainer()
        self.write_predictions(self.trainer.predict(self.dataset))

    def write_predictions(self, predictions: PredictionOutput):
        """
        Writes predictions to the file.

        :param predictions: predictions
        """
        if self.predictions is not None:
            with open(Path(self.predictions) / "results.jsonl", "w") as f:
                for prediction in self.transform_predictions(predictions):
                    print(json.dumps(prediction, ensure_ascii=False), file=f)

    @abstractmethod
    def transform_predictions(self, predictions: PredictionOutput) -> Generator[dict[str, Any], None, None]:
        """
        Transforms single sample predictions to the desired format.

        :param predictions: predictions
        :return: transformed predictions
        """
        ...

    def _init_trainer_args(self) -> TrainingArguments:
        return TrainingArguments(
            output_dir=self.predictions,
            per_device_eval_batch_size=self.per_device_batch_size
        )

    def _init_trainer(self):
        self._init_components()

        # setup batch size
        args = self._init_trainer_args()

        trainer = Trainer(
            model=self.model,
            data_collator=self.collator,
            args=args
        )
        return trainer

    def _init_components(self):
        self._init_dataset()
        self._init_model()
        self._init_tokenizer()
        self._init_collator()

    def _init_dataset(self):
        self.dataset = self.dataset_fct.create()

    def _init_model(self):
        self.model = self.model_fct.create()

    def _init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_fct.model_path, use_fast=True,
                                                       cache_dir=self.model_fct.cache_dir) \
            if self.tokenizer_fct is None else self.tokenizer_fct.create()

    def _init_collator(self):
        self.collator = DefaultDataCollator()
