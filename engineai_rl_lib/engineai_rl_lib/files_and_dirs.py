import os


def get_module_path_from_folders_in_dir(root_path, dir, prefix="", exception=""):
    item_paths = get_py_file_paths_from_dir(dir)
    import_modules = {}
    # List all items in the directory
    for item_path in item_paths:
        # Check if the item is a directory and contains an __init__.py file
        if (
            os.path.isdir(item_path)
            and "__init__.py" in os.listdir(item_path)
            and item_path.rsplit("/", 1)[-1].startswith(prefix)
        ):
            module_path = get_module_path(root_path, item_path)
            module_name = module_path.rsplit(".", 1)[-1]
            if module_name != exception:
                # Import the package using importlib
                import_modules[module_name] = module_path
    return import_modules


def get_module_path_from_files_in_dir(root_path, dir, prefix="", exception=""):
    item_paths = get_py_file_paths_from_dir(dir)
    import_modules = {}
    # List all items in the directory
    for item_path in item_paths:
        # Check if the item is a directory and contains an __init__.py file
        if item_path.endswith(".py") and item_path.rsplit("/", 1)[-1].startswith(
            prefix
        ):
            module_path = get_module_path(root_path, item_path)
            module_name = module_path.rsplit(".", 1)[-1]
            if module_name != exception:
                # Import the package using importlib
                import_modules[module_name] = module_path
    return import_modules


def get_py_file_paths_from_dir(dir, exception=[]):
    # Get the absolute path of the files
    directory_path = os.path.abspath(dir)
    item_paths = []
    for item in os.listdir(directory_path):
        if item not in exception and item.endswith(".py"):
            item_paths.append(os.path.join(directory_path, item))
    return item_paths


def get_folder_paths_from_dir(dir):
    # Get the absolute path of the directory
    directory_path = os.path.abspath(dir)
    folder_paths = []
    for item in os.listdir(directory_path):
        path = os.path.join(directory_path, item)
        if os.path.isdir(path):
            folder_paths.append(path)
    return folder_paths


def get_module_path(root_path, item_path):
    if os.path.isfile(item_path) and item_path.endswith(".py"):
        item_path_without_suffix = item_path[:-3]
    else:
        item_path_without_suffix = item_path
    module = item_path_without_suffix.replace(root_path, "")[1:]
    return module.replace("/", ".")
