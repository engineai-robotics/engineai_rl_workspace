import os
from engineai_rl_lib.files_and_dirs import get_module_path_from_files_in_dir
from engineai_gym import ENGINEAI_GYM_ROOT_DIR

file_directory = os.path.dirname(os.path.abspath(__file__))
import_modules = get_module_path_from_files_in_dir(
    ENGINEAI_GYM_ROOT_DIR, file_directory, "logger_type"
)
for module_path in import_modules.values():
    exec(f"from {module_path} import *")
