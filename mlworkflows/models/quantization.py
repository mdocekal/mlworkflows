from typing import Optional

from classconfig import ConfigurableValue
from classconfig.validators import BoolValidator, FloatValidator, AnyValidator, ListOfTypesValidator, IsNoneValidator, \
    StringValidator
from transformers import BitsAndBytesConfig


class ConfigurableBitsAndBytesFactory:
    """
    Configuration factory for bits and bytes quantization.
    """

    load_in_8bit: bool = ConfigurableValue("This flag is used to enable 8-bit quantization with LLM.int8().",
                                           user_default=False,
                                           validator=BoolValidator(),
                                           voluntary=True
                                           )
    load_in_4bit: bool = ConfigurableValue(
        "This flag is used to enable 4-bit quantization by replacing the Linear layers with FP4/NF4 layers from `bitsandbytes`.",
        user_default=False,
        validator=BoolValidator(),
        voluntary=True
        )
    llm_int8_threshold: float = ConfigurableValue(
        "This corresponds to the outlier threshold for outlier detection as described in `LLM.int8() : 8-bit Matrix Multiplication for Transformers at Scale` paper: https://arxiv.org/abs/2208.07339 Any hidden states value that is above this threshold will be considered an outlier and the operation on those values will be done in fp16. Values are usually normally distributed, that is, most values are in the range [-3.5, 3.5], but there are some exceptional systematic outliers that are very differently distributed for large models. These outliers are often in the interval [-60, -6] or [6, 60]. Int8 quantization works well for values of magnitude ~5, but beyond that, there is a significant performance penalty. A good default threshold is 6, but a lower threshold might be needed for more unstable models (small models, fine-tuning).",
        user_default=6.0,
        validator=FloatValidator(),
        voluntary=True
        )
    llm_int8_skip_modules: Optional[list[str]] = ConfigurableValue(
        "An explicit list of the modules that we do not want to convert in 8-bit. This is useful for models such as Jukebox that has several heads in different places and not necessarily at the last position. For example for `CausalLM` models, the last `lm_head` is kept in its original `dtype`.",
        validator=AnyValidator([ListOfTypesValidator(str), IsNoneValidator()]),
        voluntary=True
        )
    llm_int8_enable_fp32_cpu_offload: bool = ConfigurableValue(
        "This flag is used for advanced use cases and users that are aware of this feature. If you want to split your model in different parts and run some parts in int8 on GPU and some parts in fp32 on CPU, you can use this flag. This is useful for offloading large models such as `google/flan-t5-xxl`. Note that the int8 operations will not be run on CPU.",
        user_default=False,
        validator=BoolValidator(),
        voluntary=True
        )
    llm_int8_has_fp16_weight: bool = ConfigurableValue(
        "This flag runs LLM.int8() with 16-bit main weights. This is useful for fine-tuning as the weights do not have to be converted back and forth for the backward pass.",
        user_default=False,
        validator=BoolValidator(),
        voluntary=True)
    bnb_4bit_compute_dtype: Optional[str] = ConfigurableValue(
        "This sets the computational type which might be different than the input type. For example, inputs might be fp32, but computation can be set to bf16 for speedups.",
        user_default=None,
        voluntary=True,
        validator=AnyValidator([StringValidator(), IsNoneValidator()])
        )
    bnb_4bit_quant_type: str = ConfigurableValue(
        "This sets the quantization data type in the bnb.nn.Linear4Bit layers. Options are FP4 and NF4 data types which are specified by `fp4` or `nf4`.",
        user_default="fp4",
        voluntary=True,
        validator=StringValidator()
        )
    bnb_4bit_use_double_quant: bool = ConfigurableValue(
        "This flag is used for nested quantization where the quantization constants from the first quantization are quantized again.",
        user_default=False,
        validator=BoolValidator(),
        voluntary=True
        )
    bnb_4bit_quant_storage: Optional[str] = ConfigurableValue(
        "This sets the storage type to pack the quanitzed 4-bit prarams.",
        user_default=None,
        voluntary=True,
        validator=AnyValidator([StringValidator(), IsNoneValidator()])
        )

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def create(self) -> BitsAndBytesConfig:
        return BitsAndBytesConfig(**self.kwargs)
