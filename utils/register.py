from loguru import logger
from typing import Callable
from functools import wraps


class Register(dict):
    def __init__(self, overwritten: bool = True, *args, **kwargs):
        super(Register).__init__(*args, **kwargs)
        self._data = {}
        self.overwritten = overwritten

    def register(self, target):

        def add_item(key, value):
            if not callable(value):
                raise Exception(f"Error:{value} must be callable!")

            if key in self._data:
                if self.overwritten:
                    logger.warning(f'{value.__name__} already exists and will be overwritten!')
                    self[key] = value  # using __setitem__()
            else:
                self[key] = value  # using __setitem__()

            return value

        if callable(target):
            return add_item(target.__name__, target)  # return value
        else:
            return lambda x: add_item(target, x)  # return value

    def __call__(self, target):
        return self.register(target)

    def __getitem__(self, item):
        return self[item]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __contains__(self, key) -> bool:
        return key in self._data

    def __str__(self):
        return str(self._data)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()


register_func = Register()


# @register_func.register
@register_func
def add(a, b):
    return a + b


@register_func.register
def multiply(a, b):
    return a * b


@register_func.register('matrix multiply')
def multiply(a, b):
    pass


@register_func.register
def minus(a, b):
    return a - b


@register_func.register
def minus(a, b):
    return a - b


if __name__ == '__main__':
    for k, v in register_func.items():
        print(f"key: {k}, value: {v}")
