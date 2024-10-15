import ast
import os
import os.path
import astor
import json
from copy import deepcopy
from engineai_rl_workspace import ENGINEAI_WORKSPACE_ROOT_DIR
import re
from engineai_rl_lib.base_config import BaseConfig
from .helpers import (
    get_class_and_parent_paths,
    get_resume_path_from_original_path,
    get_last_run_path,
)


def generate_dict_from_node(node, config):
    for sub_node in node.body:
        if isinstance(sub_node, ast.Assign) and len(sub_node.targets) == 1:
            target = sub_node.targets[0]
            if isinstance(target, ast.Name):
                value = astor.to_source(sub_node.value).strip()
                try:
                    if "<expr " + target.id + ">" in config:
                        del config["<expr " + target.id + ">"]
                    eval_value = eval(value)
                    if isinstance(eval_value, dict):
                        if target.id in config:
                            del config[target.id]
                        config["<dict " + target.id + ">"] = eval_value
                    else:
                        if "<dict " + target.id + ">" in config:
                            del config["<dict " + target.id + ">"]
                        config[target.id] = eval_value
                except:
                    if target.id in config:
                        del config[target.id]
                    config["<expr " + target.id + ">"] = value
        elif isinstance(sub_node, ast.ClassDef):
            if sub_node.name not in config:
                config[sub_node.name] = {}
            generate_dict_from_node(sub_node, config[sub_node.name])


def generate_dict_from_trees(trees, config_data=None):
    if config_data is None:
        config_data = {}
    for tree in trees:
        for idx, node in enumerate(ast.walk(tree)):
            if isinstance(node, ast.ClassDef):
                generate_dict_from_node(node, config_data)
                break
    return config_data


def get_update_and_add_dict_with_missing_var_in_expression_and_del_dict(
    source_dict, target_dict, update_and_add_dict, del_dict
):
    for key, item in source_dict.items():
        if key in update_and_add_dict:
            if isinstance(item, dict) and not key.startswith("<dict"):
                del_dict[key] = {}
                get_update_and_add_dict_with_missing_var_in_expression_and_del_dict(
                    item, target_dict[key], update_and_add_dict[key], del_dict[key]
                )
                if not del_dict[key]:
                    del del_dict[key]
                if not update_and_add_dict[key]:
                    del update_and_add_dict[key]
            else:
                if key.startswith("<expr"):
                    trimmed_source_item = item.replace(" ", "").replace("\n", "")
                    trimmed_target_item = (
                        update_and_add_dict[key].replace(" ", "").replace("\n", "")
                    )
                    if trimmed_source_item == trimmed_target_item:
                        del update_and_add_dict[key]
                else:
                    if item == update_and_add_dict[key]:
                        del update_and_add_dict[key]
        else:
            if not (
                key.startswith("<dict")
                or key.startswith("<expr")
                or "<dict " + key + ">" in update_and_add_dict
                or "<expr " + key + ">" in update_and_add_dict
            ):
                del_dict[key] = item
    return update_and_add_dict, del_dict


def add_missing_var_in_expression(
    target_dict, update_and_add_dict, modified_update_and_add_dict=None
):
    if modified_update_and_add_dict is None:
        modified_update_and_add_dict = deepcopy(update_and_add_dict)
    for key, item in update_and_add_dict.items():
        if isinstance(item, dict) and not key.startswith("<dict"):
            add_missing_var_in_expression(
                target_dict[key],
                update_and_add_dict[key],
                modified_update_and_add_dict[key],
            )
        else:
            find_compliment_var(key, item, modified_update_and_add_dict, target_dict)
    return modified_update_and_add_dict


def find_compliment_var(var_name, expr, update_and_add_dict, target_dict):
    if var_name.startswith("<expr"):
        compliment_var_found = False
        for compliment_key, compliment_item in target_dict.items():
            if compliment_key.startswith("<expr"):
                trimmed_compliment_key = compliment_key[6:-1]
            else:
                trimmed_compliment_key = compliment_key
            pattern = rf"(?<![\w]){trimmed_compliment_key}(?![\w])"
            match = re.search(pattern, expr)
            if match is not None:
                update_and_add_dict[compliment_key] = compliment_item
                compliment_var_found = True
                if compliment_key.startswith("<expr"):
                    find_compliment_var(
                        compliment_key,
                        compliment_item,
                        update_and_add_dict,
                        target_dict,
                    )
        if not compliment_var_found:
            raise ValueError(
                f'Variables in expression "{expr}" not found'.format(var_name)
            )


def get_update_and_add_dict_and_del_dict(
    source_dict, target_dict, update_and_add_dict=None, del_dict=None
):
    if update_and_add_dict is None:
        update_and_add_dict = deepcopy(target_dict)
    if del_dict is None:
        del_dict = {}
    (
        update_and_add_dict,
        del_dict,
    ) = get_update_and_add_dict_with_missing_var_in_expression_and_del_dict(
        source_dict, target_dict, update_and_add_dict, del_dict
    )
    update_and_add_dict = add_missing_var_in_expression(
        target_dict, update_and_add_dict
    )
    return update_and_add_dict, del_dict


def change_class_attributes_value(
    pyfile_name,
    is_root_node,
    node,
    updates,
    changes_made,
    expression=False,
    key_names=None,
):
    if is_root_node:
        section_updates = updates
        key_names = node.name
    else:
        if node.name in updates:
            section_updates = updates[node.name]
        else:
            return
    for sub_node in node.body:
        if isinstance(sub_node, ast.Assign) and len(sub_node.targets) == 1:
            target = sub_node.targets[0]
            if isinstance(target, ast.Name):
                if (
                    target.id in section_updates
                    or "<expr " + target.id + ">" in section_updates
                    or "<dict " + target.id + ">" in section_updates
                ):
                    try:
                        old_value = eval(astor.to_source(sub_node.value).strip())
                    except:
                        old_value = astor.to_source(sub_node.value).strip()

                    if target.id in section_updates:
                        target_name = target.id
                    elif "<expr " + target.id + ">" in section_updates:
                        target_name = "<expr " + target.id + ">"
                    elif "<dict " + target.id + ">" in section_updates:
                        target_name = "<dict " + target.id + ">"
                    new_value = section_updates[target_name]
                    attr_name = key_names + "." + target.id
                    if not ("<expr" in target_name and not expression):
                        if old_value != new_value:
                            print(
                                f"Updating attribute: {attr_name}: {old_value} -> {new_value} (in {pyfile_name})"
                            )
                            if target_name.startswith("<expr"):
                                sub_node.value = (
                                    ast.parse(eval(repr(new_value))).body[0].value
                                )
                            else:
                                sub_node.value = (
                                    ast.parse(repr(new_value)).body[0].value
                                )
                            changes_made[0] = True
                            del section_updates[target_name]
        elif isinstance(sub_node, ast.ClassDef):
            change_class_attributes_value(
                pyfile_name,
                False,
                sub_node,
                section_updates,
                changes_made,
                expression,
                key_names + "." + sub_node.name,
            )
            if sub_node.name in section_updates:
                if not section_updates[sub_node.name]:
                    del section_updates[sub_node.name]


def add_class_attributes_value(
    pyfile_name,
    source_config,
    is_root_node,
    node,
    updates,
    expression=True,
    key_names=None,
    parent_node_name=None,
):
    if is_root_node:
        section_updates = updates
        key_names = node.name
        parent_node_name = node.bases[0].id
    else:
        if node.name in updates:
            section_updates = updates[node.name]
        else:
            return

    # Track existing classes
    existing_classes = {
        sub_node.name: sub_node
        for sub_node in node.body
        if isinstance(sub_node, ast.ClassDef)
    }
    added_attr_names = []
    # Add missing attributes or update existing classes
    for attr_name, attr_value in section_updates.items():
        if isinstance(attr_value, dict) and not attr_name.startswith("<dict "):
            if attr_name in existing_classes:
                # Recursively add attributes to the existing nested class
                if attr_name in source_config:
                    new_source_config = source_config[attr_name]
                    new_parent_node_name = parent_node_name + "." + attr_name
                else:
                    new_source_config = {}
                    new_parent_node_name = None
                add_class_attributes_value(
                    pyfile_name,
                    new_source_config,
                    False,
                    existing_classes[attr_name],
                    {attr_name: attr_value},
                    expression,
                    key_names + "." + attr_name,
                    new_parent_node_name,
                )

            else:
                # Create a new nested class
                filtered_attr_name = (
                    attr_name[6:-1] if attr_name.startswith("<dict ") else attr_name
                )
                if attr_name in source_config:
                    new_parent_node_name = parent_node_name + "." + filtered_attr_name
                    bases = [new_parent_node_name]
                    new_source_config = source_config[attr_name]
                else:
                    new_parent_node_name = None
                    bases = []
                    new_source_config = {}

                nested_class = ast.ClassDef(
                    name=filtered_attr_name,
                    bases=bases,
                    keywords=[],
                    body=[],
                    decorator_list=[],
                )

                # Recursively add attributes to the nested class
                add_class_attributes_value(
                    pyfile_name,
                    new_source_config,
                    False,
                    nested_class,
                    {attr_name: attr_value},
                    expression,
                    key_names + "." + filtered_attr_name,
                    new_parent_node_name,
                )

                node.body.append(nested_class)
                added_attr_names.append(attr_name)
        else:
            # Check if the attribute already exists
            if not any(
                isinstance(sub_node, ast.Assign) and sub_node.targets[0].id == attr_name
                for sub_node in node.body
            ):
                new_value = None
                filtered_attr_name = (
                    attr_name[6:-1]
                    if attr_name.startswith("<dict ") or attr_name.startswith("<expr")
                    else attr_name
                )
                if attr_name.startswith("<expr"):
                    if expression:
                        new_value = ast.parse(eval(repr(attr_value))).body[0].value
                elif attr_name.startswith("<dict "):
                    new_value = ast.parse(repr(attr_value)).body[0].value
                else:
                    new_value = ast.parse(repr(attr_value)).body[0].value
                if new_value is not None:
                    new_assign = ast.Assign(
                        targets=[ast.Name(id=filtered_attr_name, ctx=ast.Store())],
                        value=new_value,
                    )
                    printed_attr_name = key_names + "." + filtered_attr_name
                    if attr_name in source_config:
                        print(
                            f"Updating attribute: {printed_attr_name} = {source_config[attr_name]} -> {attr_value} (in {pyfile_name})"
                        )
                    else:
                        print(
                            f"Adding missing attribute: {printed_attr_name} = {attr_value} (in {pyfile_name})"
                        )
                    node.body.append(new_assign)
                    added_attr_names.append(attr_name)
    for added_attr_name in added_attr_names:
        del section_updates[added_attr_name]


def delete_class_attributes_value(pyfile_name, node, del_items, key_names=None):
    if key_names is None:
        key_names = node.name
    for attr_name, attr_value in del_items.items():
        if isinstance(attr_value, dict):
            for sub_node in node.body:
                if hasattr(sub_node, "name"):
                    if sub_node.name == attr_name:
                        delete_class_attributes_value(
                            pyfile_name,
                            sub_node,
                            del_items[attr_name],
                            key_names + "." + attr_name,
                        )
                        if not sub_node.body:
                            node.body.remove(sub_node)
        else:
            for sub_node in node.body:
                if isinstance(sub_node, ast.Assign) and len(sub_node.targets) == 1:
                    if sub_node.targets[0].id == attr_name:
                        node.body.remove(sub_node)
            attr_name = key_names + "." + attr_name
            print(f"Deleting redundant attribute: {attr_name} (in {pyfile_name})")


def generate_py_from_dict(target_config_data, py_files, resume_dir):
    trees = get_trees_from_py_files(py_files)
    source_config_data = generate_dict_from_trees(trees)
    trees.reverse()
    for idx, py_file in enumerate(py_files):
        py_files[idx] = get_resume_path_from_original_path(py_file, resume_dir)
    update_and_add_dict, del_dict = get_update_and_add_dict_and_del_dict(
        source_config_data, target_config_data
    )
    changes_made = [False]
    for idx, node in enumerate(ast.walk(trees[0])):
        if isinstance(node, ast.ClassDef):
            change_class_attributes_value(
                py_files[0], True, node, update_and_add_dict, changes_made, False
            )
            change_class_attributes_value(
                py_files[0], True, node, update_and_add_dict, changes_made, True
            )
            break
    for idx, node in enumerate(ast.walk(trees[0])):
        if isinstance(node, ast.ClassDef):
            add_class_attributes_value(
                py_files[0], source_config_data, True, node, update_and_add_dict, False
            )
            add_class_attributes_value(
                py_files[0], source_config_data, True, node, update_and_add_dict, True
            )
            break
    for tree, py_file in zip(trees, py_files):
        for idx, node in enumerate(ast.walk(tree)):
            if isinstance(node, ast.ClassDef):
                delete_class_attributes_value(py_file, node, del_dict)
                break
    for tree, py_file in zip(trees, py_files):
        save_file = os.path.join(resume_dir, py_file)
        save_dir = save_file.rsplit("/", 1)[0]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(save_file + ".txt", "w") as file:
            file.write(astor.to_source(tree))


def get_trees_from_py_files(py_files):
    trees = []
    for idx, py_file in enumerate(py_files[::-1]):
        with open(py_file) as file:
            python_content = file.read()
        trees.append(ast.parse(python_content))
    return trees


def generate_resume_cfg_files_from_json(args, log_dir=None):
    from .exp_registry import exp_registry
    import engineai_rl_workspace.exps

    if log_dir is None:
        if args.resume or args.run_exist:
            if args.log_root is not None:
                log_root = args.log_root
            else:
                log_root = os.path.join(
                    ENGINEAI_WORKSPACE_ROOT_DIR,
                    "logs",
                    args.exp_name,
                    args.sub_exp_name,
                )
            if args.load_run == -1:
                load_run = get_last_run_path(log_root)
            else:
                load_run = args.load_run
            log_dir = os.path.join(log_root, load_run)
        else:
            raise RuntimeError("Current log dir is not provided!")
    resume_dir = os.path.join(log_dir, "resume")
    json_file_path = os.path.join(log_dir, "config.json")
    with open(json_file_path) as file:
        config_data = json.load(file)
    env_cfg_items = get_class_and_parent_paths(
        type(exp_registry.env_cfgs[args.exp_name]), BaseConfig
    )
    train_cfg_items = get_class_and_parent_paths(
        type(exp_registry.algo_cfgs[args.exp_name]), BaseConfig
    )
    generate_py_from_dict(config_data["env_cfg"], env_cfg_items, resume_dir)
    generate_py_from_dict(config_data["algo_cfg"], train_cfg_items, resume_dir)


def get_dict_from_cfg_before_modification(cfg):
    cfg_items = get_class_and_parent_paths(type(cfg), BaseConfig)
    trees = get_trees_from_py_files(cfg_items)
    return generate_dict_from_trees(trees)


def update_cfg_dict_from_args(cfg, args):
    if "env_cfg" in cfg:
        if args.num_envs is not None:
            cfg["env_cfg"]["env"]["num_envs"] = args.num_envs
    if "algo_cfg" in cfg:
        if args.seed is not None:
            cfg["algo_cfg"]["seed"] = args.seed
        if args.max_iterations is not None:
            cfg["algo_cfg"]["runner"]["max_iterations"] = args.max_iterations
