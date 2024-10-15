import re
import os, sys
import importlib
import importlib.util


def add_space_to_class_name(class_name):
    def add_space(match):
        # Return the matched string with spaces added before each uppercase letter (except the first)
        return re.sub(r"(?<!^)(?=[A-Z])", " ", match.group(0))

    return re.sub(r"\b[A-Z][a-zA-Z]*", add_space, class_name)


def class_name_to_instance_name(class_name):
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", class_name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def instance_name_to_class_name(instance_name):
    # Split the instance name by underscores
    words = instance_name.split("_")
    # Capitalize the first letter of each word and join them
    class_name = "".join(word.capitalize() for word in words)
    return class_name


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def get_classes_of_base_from_files_with_prefix_in_folder(
    dir_path, file_prefix, base_class
):
    classes = []
    for filename in os.listdir(dir_path):
        if filename.startswith(file_prefix) and filename.endswith(".py"):
            module_name = filename[:-3]  # remove the .py to get the module name
            spec = importlib.util.spec_from_file_location(
                module_name, os.path.join(dir_path, filename)
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            for attr_name in dir(module):
                attr_value = getattr(module, attr_name)
                if (
                    isinstance(attr_value, type)
                    and issubclass(attr_value, base_class)
                    and attr_name != base_class.__name__
                ):
                    classes.append(attr_value)
    return classes


def get_class_from_file(file_path, name):
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return eval(f"module.{name}")


def add_instance_properties_and_methods_to_class(original_instance, target_instance):
    # Store original instance for property access
    target_instance._original = original_instance

    # Dynamically copy properties
    for name in dir(original_instance):
        attr = getattr(type(original_instance), name, None)
        if isinstance(attr, property) and not name.startswith("_"):
            # Create property that delegates to original instance
            new_prop = property(
                lambda self, name=name: getattr(self._original, name),
                lambda self, value, name=name: setattr(self._original, name, value),
                lambda self, name=name: delattr(self._original, name),
                doc=attr.__doc__,
            )
            setattr(type(target_instance), name, new_prop)

    # Dynamically copy methods
    for name in dir(original_instance):
        if (
            callable(getattr(original_instance, name))
            and not name.startswith("__")
            and not hasattr(target_instance, name)
        ):
            method = getattr(original_instance, name)

            # Create a method that delegates to the original instance
            def make_method(name=name):
                def delegated_method(self, *args, **kwargs):
                    return getattr(self._original, name)(*args, **kwargs)

                return delegated_method

            bound_method = make_method(name)
            setattr(type(target_instance), name, bound_method)
