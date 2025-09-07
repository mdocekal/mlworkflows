from typing import Optional

from classconfig import ConfigurableValue, RelativePathTransformer
from classconfig.validators import AnyValidator, IsNoneValidator, StringValidator, BoolValidator
from transformers import EvalPrediction
import evaluate


class ComputeMetrics:
    """
    Functor for computing metrics that is compatible with Trainer
    """

    path: str = ConfigurableValue("Path to the evaluation processing script with the evaluation builder. Can be either: - a local path to processing script or the directory containing the script (if the script has the same name as the directory), e.g. `'./metrics/rouge'` or `'./metrics/rouge/rouge.py'` - a evaluation module identifier on the HuggingFace evaluate repo e.g. `'rouge'` or `'bleu'` that are in either `'metrics/'`,`'comparisons/'`, or `'measurements/'` depending on the provided `module_type`",
                                  transform=RelativePathTransformer(force_relative_prefix=True))
    config_name: Optional[str] = ConfigurableValue("Selecting a configuration for the metric (e.g. the GLUE metric has a configuration for each subset).",
                                         user_default=None, validator=AnyValidator([IsNoneValidator(), StringValidator()]))
    module_type: str = ConfigurableValue("Type of evaluation module, can be one of `'metric'`, `'comparison'`, or `'measurement'`.",
                                            user_default='metric', validator=StringValidator())
    keep_in_memory: bool = ConfigurableValue("Whether to store the temporary results in memory (defaults to `False`).",
                                                user_default=False, validator=BoolValidator())
    revision: Optional[str] = ConfigurableValue("If specified, the module will be loaded from the datasets repository at this version. By default it is set to the local version of the lib. Specifying a version that is different from your local version of the lib might cause compatibility issues.",
                                                user_default=None, validator=AnyValidator([IsNoneValidator(), StringValidator()]))
    config: Optional[dict] = ConfigurableValue("Additional configuration for the evaluation module.",
                                        user_default=None, validator=AnyValidator([dict, IsNoneValidator()]))

    def __init__(self, path: str, config_name: Optional[str] = None, module_type: str = 'metric',
                 keep_in_memory: bool = False, revision: Optional[str] = None, config: Optional[dict] = None, **kwargs):

        if config is None:
            config = {}

        config.update(kwargs)

        self.evaluator = evaluate.load(path,
                                       config_name=config_name,
                                       module_type=module_type,
                                       keep_in_memory=keep_in_memory,
                                       revision=revision,
                                       **config)

        self.transform = None

    def __call__(self, eval_preds: EvalPrediction, compute_result: bool = True) -> Optional[dict]:
        """
        Computes evaluation metrics.

        :param eval_preds: Evaluation predictions.
        :param compute_result: Whether to compute the results.
        :return: Metrics.
        """
        if self.transform is not None:
            self.transform(eval_preds)

        self.evaluator.add_batch(
            predictions=eval_preds.predictions,
            references=eval_preds.label_ids
        )
        if compute_result:
            return self.evaluator.compute()
        return None

