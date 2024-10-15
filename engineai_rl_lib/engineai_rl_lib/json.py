import os
import json


def convert_to_json(target_dict, output_file_name):
    converted_json = json.dumps(
        target_dict,
        indent=4,
        default=lambda o: None
        if isinstance(o, type(None))
        else (
            f"<<non-serializable: {type(o)}>>"
            if not isinstance(o, (int, float, str, bool, list, dict))
            else str(o)
        ),
    )

    with open(output_file_name, "w") as json_file:
        json_file.write(converted_json)


def save_json_files(target_dict, log_dir, filename):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    convert_to_json(target_dict, os.path.join(log_dir, filename))
