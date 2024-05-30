import json


def demon_prompt_generate(demon_file_path, demon_parameter):
    with open(demon_file_path, 'r', encoding='utf-8') as demon_file:
        demon_list = demon_file.readlines()
    demon_instruction = ''
    demon_key_list = demon_parameter['key']
    demon_template = demon_parameter['template']
    for demon in demon_list:
        demon = json.loads(demon)
        demon_value_list = [demon[key] for key in demon_key_list]
        demon_instruction += str(demon_template).format(*demon_value_list)

    return demon_instruction, len(demon_list)


def bbh_demon_prompt_generate(demon_file_path):
    with open(demon_file_path, 'r', encoding='utf-8') as demon_file:
        demon_instruction = demon_file.read()

        return demon_instruction, 3


def task_instruction_generate(jsonobj, instruction_parameter):
    key_list = instruction_parameter['key']
    template = instruction_parameter['template']
    value_list = [jsonobj[key] for key in key_list]
    instruction = str(template).format(*value_list)

    return instruction
