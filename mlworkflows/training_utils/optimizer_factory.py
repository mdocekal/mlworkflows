from abc import ABC, abstractmethod
from typing import Union, Iterable, Dict, Type

import torch
from bitsandbytes.optim import Lion
from classconfig import ConfigurableValue
from classconfig.validators import ValueInIntervalFloatValidator, FloatValidator, BoolValidator, \
    ValueInIntervalIntegerValidator
from torch.optim import SGD, Adam, AdamW


class OptimizerFactory(ABC):
    """
    Abstract base class for optimizers creation.
    """

    @abstractmethod
    def create(self, module: torch.nn.Module) -> torch.optim.Optimizer:
        """
        Creates optimizer for given module.

        :param module: Module which parameters you want to optimize.
        :return: Created optimizer for given module and with settings that are hold by factory.
        """
        pass

    @abstractmethod
    def create_for_params(self, params: Union[Iterable[torch.Tensor], Iterable[Dict]]) -> torch.optim.Optimizer:
        """
        Creates optimizer for given parameters.

        :param params: Parameters that should be optimized.
            An iterable of torch.Tensors or dicts which specifies which Tensors should be optimized along with group
            specific optimization options.
                Example of groups:
                    [
                        {'params': ..., 'weight_decay': ...},
                        {'params': ..., 'weight_decay': ...}
                    ]
        :return: Created optimizer for given params and with settings that are hold by factory.
        """
        pass

    @property
    @abstractmethod
    def optimizer_cls(self) -> Type[torch.optim.Optimizer]:
        """
        Returns the class of optimizer that is created by this factory.
        """
        pass

    @property
    @abstractmethod
    def optimizer_kwargs(self) -> Dict:
        """
        Returns the kwargs that are used to create the optimizer.
        """
        pass


class AnyOptimizerFactory(OptimizerFactory):
    """
    Class that allows creation of any optimizer on demand.
    """

    def __init__(self, opt_cls: Type[torch.optim.Optimizer], attr: Dict, params_attr: str = "params"):
        """
        Initialization of factory.

        :param opt_cls: Optimizer class that should be used.
        :param attr: Dictionary with attributes that should be used. Beware that the attribute with name paramsAttr
            is reserved for model parameters
        :param params_attr: Name of attribute that will be used to pass model parameters to optimizer.
        """

        self.opt_cls = opt_cls
        self.attr = attr
        self.params_attr = params_attr

    def create(self, module: torch.nn.Module) -> torch.optim.Optimizer:
        return self.create_for_params(module.parameters())

    def create_for_params(self, params: Union[Iterable[torch.Tensor], Iterable[Dict]]) -> torch.optim.Optimizer:
        self.attr[self.params_attr] = params
        return self.opt_cls(**self.attr)

    @property
    def optimizer_cls(self) -> Type[torch.optim.Optimizer]:
        return self.opt_cls

    @property
    def optimizer_kwargs(self) -> Dict:
        return self.attr


class SGDFactory(AnyOptimizerFactory):
    """
    Factory for SGD optimizer.
        https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
    """
    learning_rate: float = ConfigurableValue("learning rate", validator=ValueInIntervalFloatValidator(0,
                                                                                                      left_inclusive=False))
    momentum: float = ConfigurableValue("momentum factor", user_default=0, validator=FloatValidator())
    weight_decay: float = ConfigurableValue("weight decay (L2 penalty)", user_default=0, validator=FloatValidator())
    dampening: float = ConfigurableValue("dampening for momentum", user_default=0, validator=FloatValidator())
    nesterov: bool = ConfigurableValue("enables Nesterov momentum", user_default=False,
                                       validator=BoolValidator())
    maximize: bool = ConfigurableValue("maximize the params based on the objective, instead of minimizing",
                                       user_default=False, validator=BoolValidator())

    def __init__(self, **args):
        super().__init__(SGD, {
            "lr": args["learning_rate"],
            "momentum": args["momentum"],
            "weight_decay": args["weight_decay"],
            "dampening": args["dampening"],
            "nesterov": args["nesterov"],
            "maximize": args["maximize"]
        })


class AdamFactory(AnyOptimizerFactory):
    """
    Factory for Adam optimizer.
        https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
    """

    learning_rate: float = ConfigurableValue("learning rate", validator=ValueInIntervalFloatValidator(0,
                                                                                                      left_inclusive=False))
    beta_1: float = ConfigurableValue("coefficient used for computing running averages of gradient and its square",
                                      user_default=0.9,
                                      validator=FloatValidator())
    beta_2: float = ConfigurableValue("coefficient used for computing running averages of gradient and its square",
                                      user_default=0.999,
                                      validator=FloatValidator())
    eps: float = ConfigurableValue("term added to the denominator to improve numerical stability",
                                      user_default=1e-8,
                                      validator=FloatValidator())
    weight_decay: float = ConfigurableValue("weight decay (L2 penalty)", user_default=0, validator=FloatValidator())
    amsgrad: bool = ConfigurableValue("whether to use the AMSGrad variant of this algorithm from the paper: "
                                      "On the Convergence of Adam and Beyond",
                                      user_default=False, validator=BoolValidator())
    maximize: bool = ConfigurableValue("maximize the params based on the objective, instead of minimizing",
                                       user_default=False, validator=BoolValidator())

    def __init__(self, **args):
        super().__init__(Adam, {
            "lr": args["learning_rate"],
            "betas": (args["beta_1"], args["beta_2"]),
            "eps": args["eps"],
            "weight_decay": args["weight_decay"],
            "amsgrad": args["amsgrad"],
            "maximize": args["maximize"]
        })


class AdamWFactory(AnyOptimizerFactory):
    """
    Factory for Adam optimizer.
        https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW
    """

    learning_rate: float = ConfigurableValue("learning rate", validator=ValueInIntervalFloatValidator(0,
                                                                                                      left_inclusive=False))
    beta_1: float = ConfigurableValue("coefficient used for computing running averages of gradient and its square",
                                      user_default=0.9,
                                      validator=FloatValidator())
    beta_2: float = ConfigurableValue("coefficient used for computing running averages of gradient and its square",
                                      user_default=0.999,
                                      validator=FloatValidator())
    eps: float = ConfigurableValue("term added to the denominator to improve numerical stability",
                                      user_default=1e-8,
                                      validator=FloatValidator())
    weight_decay: float = ConfigurableValue("weight decay coefficient", user_default=1e-2, validator=FloatValidator())
    amsgrad: bool = ConfigurableValue("whether to use the AMSGrad variant of this algorithm from the paper: "
                                      "On the Convergence of Adam and Beyond",
                                      user_default=False, validator=BoolValidator())

    eight_bit: bool = ConfigurableValue("whether to use 8 bit precision", user_default=False, validator=BoolValidator())

    min_8bit_size: int = ConfigurableValue("The minimum number of elements of the parameter tensors for 8-bit optimization.",
                                           user_default=4096, validator=ValueInIntervalIntegerValidator(0))
    percentile_clipping: int = ConfigurableValue("Adapts clipping threshold automatically by tracking the last 100 gradient norms and clipping the gradient at a certain percentile to improve stability.",
                                                 user_default=100, validator=ValueInIntervalIntegerValidator(0, 100))
    block_wise: bool = ConfigurableValue("Whether to independently quantize each block of tensors to reduce outlier effects and improve stability.",
                                         user_default=True, validator=BoolValidator())
    is_paged: bool = ConfigurableValue("WWhether the optimizer is a paged optimizer or not.", user_default=False, validator=BoolValidator())

    def __init__(self, **args):
        super().__init__(AdamW, {
            "lr": args["learning_rate"],
            "betas": (args["beta_1"], args["beta_2"]),
            "eps": args["eps"],
            "weight_decay": args["weight_decay"],
            "amsgrad": args["amsgrad"],
            "optim_bits": 8 if args["eight_bit"] else 32,
            "min_8bit_size": args["min_8bit_size"],
            "percentile_clipping": args["percentile_clipping"],
            "block_wise": args["block_wise"],
            "is_paged": args["is_paged"]
        })


class LionFactory(AnyOptimizerFactory):
    """
    Factory for Lion optimizer:
        https://github.com/lucidrains/lion-pytorch
        https://arxiv.org/abs/2302.06675
    """

    learning_rate: float = ConfigurableValue("learning rate", validator=ValueInIntervalFloatValidator(0,
                                                                                                      left_inclusive=False))
    beta_1: float = ConfigurableValue("coefficient used for momentum tacking",
                                      user_default=0.9,
                                      validator=FloatValidator())
    beta_2: float = ConfigurableValue("coefficient used for momentum tacking",
                                      user_default=0.99,
                                      validator=FloatValidator())
    weight_decay: float = ConfigurableValue("weight decay coefficient", user_default=0.0, validator=FloatValidator())

    eight_bit: bool = ConfigurableValue("whether to use 8 bit precision", user_default=False, validator=BoolValidator())

    min_8bit_size: int = ConfigurableValue("The minimum number of elements of the parameter tensors for 8-bit optimization.",
                                           user_default=4096, validator=ValueInIntervalIntegerValidator(0))
    percentile_clipping: int = ConfigurableValue("Adapts clipping threshold automatically by tracking the last 100 gradient norms and clipping the gradient at a certain percentile to improve stability.",
                                                 user_default=100, validator=ValueInIntervalIntegerValidator(0, 100))
    block_wise: bool = ConfigurableValue("Whether to independently quantize each block of tensors to reduce outlier effects and improve stability.",
                                         user_default=True, validator=BoolValidator())
    is_paged: bool = ConfigurableValue("WWhether the optimizer is a paged optimizer or not.", user_default=False, validator=BoolValidator())

    def __init__(self, **args):
        opt_cls = Lion
        super().__init__(opt_cls, {
            "lr": args["learning_rate"],
            "betas": (args["beta_1"], args["beta_2"]),
            "weight_decay": args["weight_decay"],
            "optim_bits": 8 if args["eight_bit"] else 32,
            "min_8bit_size": args["min_8bit_size"],
            "percentile_clipping": args["percentile_clipping"],
            "block_wise": args["block_wise"],
            "is_paged": args["is_paged"]
        })
