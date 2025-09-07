import json

import jinja2
from classconfig import ConfigurableValue


class Jinja2EnvironmentSingletonFactory:
    """
    Singleton factory for Jinja2 environment.
    It is here to be able to add additional filters.
    """
    jinja_env: jinja2.Environment = None

    def __init__(self):
        if not self.jinja_env:
            self.jinja_env = jinja2.Environment()
            # default tojson doesn't allow to use all the arguments of json.dumps
            self.jinja_env.filters["tojson"] = json.dumps
            self.jinja_env.filters["filter_dict"] = lambda d, keys: {k: v for k, v in d.items() if k in keys}


class Template:
    template: str = ConfigurableValue("Jinja2 template")

    def __init__(self, template: str):
        self.template = template
        self.jinja = Jinja2EnvironmentSingletonFactory().jinja_env
        self.jinja_template = self.jinja.from_string(template)

    def render(self, data: dict[str, any]) -> str:
        """
        Renders the template with the data.

        :param data: data
        :return: rendered template
        """
        return self.jinja_template.render(data)
