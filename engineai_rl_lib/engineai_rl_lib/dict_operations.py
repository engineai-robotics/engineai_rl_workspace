def convert_dict(target_dict, pre_key=""):
    converted_dict = {}
    tmp_converted_dict = {}
    for key, value in target_dict.items():
        if isinstance(value, dict):
            tmp_converted_dict.update(convert_dict(value, key))
        else:
            if pre_key:
                converted_dict[f"{pre_key}/{key}"] = value
            else:
                converted_dict[key] = value
    for key, value in tmp_converted_dict.items():
        if pre_key:
            converted_dict[f"{pre_key}/{key}"] = value
        else:
            converted_dict[key] = value

    return converted_dict


def convert_dicts(target_dicts):
    converted_dict_list = []
    for target_dict in target_dicts:
        converted_dict_list.append(convert_dict(target_dict))
    return converted_dict_list
