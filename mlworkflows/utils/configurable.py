from typing import Type, Optional, Any, Dict, Union, AbstractSet

from classconfig import ConfigurableSubclassFactory, LoadedConfig, DelayedFactory, T, ConfigurableFactory
from transformers import AutoConfig


class HFModelConfig(ConfigurableSubclassFactory):
    """
    Factory that allows to automatically wrap config class of a hugging face model.
    """

    def __init__(self, desc: Optional[str] = None, name: Optional[str] = None,
                 user_default: Any = None, delay_init: bool = False, voluntary: bool = False,
                 file_override_user_defaults: Optional[Dict] = None,
                 omit: Optional[Dict[str, Union[AbstractSet, Dict]]] = None):
        """
        defines for which class this factory is for

        :param desc: description of this attribute for user
        :param name: name of attribute that is suitable for user
            If None it will be obtained with __set_name__ method
        :param user_default: default class that will be shown to user
        :param delay_init: delays class initializations
            It means that the :method:`~ConfigurableFactory.create` returns just a factory with loaded configuration.
            Might be useful when a class could be "initialized" just partially with configuration
        :param voluntary: If True this value might be missing in config and the default value will be used.
        :param file_override_user_defaults: overrides default values of configurable attributes for generating
            configuration file. It is useful when the default value is not suitable for user in different context.
            It is not overriding the user_default attribute of configurable attribute.
        :param omit: omit these attributes from configuration file as it is expected that they will be provided
            programmatically for this reason the delay_init will be set to True if not None

            Use "" to address attributes that are directly in the class, otherwise use the name of the attribute to
            address attributes that are in the class that is defined in the attribute. Works recursively.

            E.g.:
            {
                "": {"batch_size"}
                "model": {"input_size"}
            }
            Will omit batch_size from the passed class and input_size attribute of model attribute.

        """

        super().__init__(desc, name, voluntary)
        self.parent_cls_type = parent_cls_type
        self.delay_init = delay_init or omit is not None
        self.user_default = user_default
        self.file_override_user_defaults = file_override_user_defaults
        self.omit = omit

    def create(self, config: LoadedConfig[str, Any], omit: Optional[Dict[str, Union[AbstractSet, Dict]]] = None) \
            -> Union[T, DelayedFactory[T]]:
        """
        Creates instance of given class.

        :param config: configuration for initialization
            It expects dictionary with:
                cls: defining class name
                config: defining class configuration
        :param omit: omit these attributes when creating
            if not None the delay_init will be used
        :return: initialized class or factory for delayed initialization of delay_init is active
        """

        if self.omit is not None and omit is not None:
            act_omit = self.omit | omit
        else:
            act_omit = self.omit if self.omit is not None else omit

        config = AutoConfig.from_pretrained(config["cls"], **config["config"])
        factory = ConfigurableFactory(sub_cls_from_its_name(self.parent_cls_type, config["cls"]), None, None,
                                      self.delay_init, self.voluntary, self.file_override_user_defaults, act_omit)
        return factory.create(config["config"])
