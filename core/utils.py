from configuration import Config


def get_num_classes_and_blank_index():
    num_classes = len(Config.get_idx2char())
    blank_index = num_classes - 1
    return num_classes, blank_index


def index_to_char(inputs, idx2char_dict, blank_index, merge_repeated=False):
    chars = []
    for item in inputs:
        text = ""
        pre_char = -1
        for current_char in item:
            if merge_repeated:
                if current_char == pre_char:
                    continue
            pre_char = current_char
            if current_char == blank_index:
                continue
            text += idx2char_dict[current_char]
        chars.append(text)
    return chars