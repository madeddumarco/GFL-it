from nltk import WordPunctTokenizer
import csv
import Levenshtein

tokenizer = WordPunctTokenizer()

SIMILARITY_LOOKUP = {}

def nltk_tokenize(text: str) -> list[str]:
    return tokenizer.tokenize(text)



def merge_slash_tokens(tokens: list[str]) -> list[str]:
    """Merge sequences like ['word', '/', 'word'] into ['word/word'],
    or into a single 'word' if all parts in the chain are identical."""
    result = []
    i = 0
    n = len(tokens)
    while i < n:
        if i + 2 < n and tokens[i + 1] == '/':
            parts = [tokens[i]]
            i += 1
            while i + 1 < n and tokens[i] == '/':
                parts.append(tokens[i + 1])
                i += 2
            if len(set(parts)) == 1:
                result.append(parts[0])
            else:
                result.append('/'.join(parts))
        else:
            result.append(tokens[i])
            i += 1
    return result


def get_csv_token(token: dict, index_token: int) -> dict:
    return {
        f"span_token_{index_token}_text": token.token,
        f"span_token_{index_token}_forms_cv": [str(f) for f in token.forms_cv],
        f"span_token_{index_token}_forms_neo": [str(f) for f in token.forms_neo],
    }

def get_csv_tokens(list_tokens: list[dict]) -> list[dict]:
    # one dict per index, NOT merged
    return [get_csv_token(tok, i) for i, tok in enumerate(list_tokens)]

def split_list_for_csv(list_elem: list|str, prefix: str) -> list[dict]:
    # one dict per index, NOT merged
    if isinstance(list_elem, str): #this is for diff_tokens
        list_elem = [list_elem]
    return [{f"{prefix}_{i}": elem} for i, elem in enumerate(list_elem)]

def interleave_indexed_dicts(*lists_of_dicts: list[dict]) -> dict:
    """
    Takes several lists of per-index dicts (possibly different lengths)
    and merges them so all fields for index i come before index i+1.
    """
    max_len = max((len(l) for l in lists_of_dicts), default=0)
    merged = {}
    for i in range(max_len):
        for lst in lists_of_dicts:
            if i < len(lst):
                merged.update(lst[i])
    return merged


import re

def _field_sort_key(key, base_fields):
    if key in base_fields:
        return (-1, base_fields.index(key), "")

    m = re.match(r"(span_token_|prediction_token_|evaluation_token_)(\d+)(.*)", key)
    if m:
        prefix, idx, rest = m.groups()
        group_order = {
            "span_token_": 0,
            "prediction_token_": 1,
            "evaluation_token_": 2,
        }[prefix]
        return (int(idx), group_order, rest)

    # fallback for anything unexpected
    return (10**9, 0, key)

def write_csv_samples(list_of_dics, name_file):
    if len(list_of_dics) > 0:
        base_fields = ["id_sentence", "sentence", "id_span", "span", "output"]

        all_keys = set()
        for d in list_of_dics:
            all_keys.update(d.keys())

        fieldnames = sorted(all_keys, key=lambda k: _field_sort_key(k, base_fields))

        with open(name_file, 'w', newline='', encoding="utf-8") as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames, extrasaction="ignore")
            dict_writer.writeheader()
            dict_writer.writerows(list_of_dics)

def _token_similarity(tok1: str, tok2: str) -> float:
    """Compute a simple similarity score between two tokens based on character overlap."""
    tok1 = tok1.casefold()
    tok2 = tok2.casefold()
    #calculates levensthein distance, updates the lookup table

    key = (tok1, tok2) if tok1>tok2 else (tok2, tok1)
    if key in SIMILARITY_LOOKUP:
        return SIMILARITY_LOOKUP[key]
    else:
        distance = Levenshtein.distance(tok1, tok2)
        max_len = max(len(tok1), len(tok2))
        similarity = 1 - (distance / max_len) if max_len > 0 else 1.0
        SIMILARITY_LOOKUP[key] = similarity
        return similarity

def get_most_similar_token(tok: str, list_compare: list[str]) -> tuple[int, float]:
    """given a tok and a list to compare it with, it returns the index and similarity of the most similar token string"""
    max_similarity = -1
    best_index = 0
    for i, compare_tok in enumerate(list_compare):
        if "/" in compare_tok:
            list_single_tokens = compare_tok.split("/")
        else:
            list_single_tokens = [compare_tok]
        
        for single_token in list_single_tokens:
            similarity = _token_similarity(tok, single_token)
            if similarity > max_similarity:
                max_similarity = similarity
                best_index = i
    return (best_index, max_similarity)

def pad_list(list_to_pad: list[str],
             start_actual_list: int,
             len_to_get: int) -> list[str]:
    pad_str = "<error>"
    pad_sx = [pad_str] * start_actual_list
    new_list = pad_sx + list_to_pad
    pad_dx = [pad_str] * (len_to_get-len(new_list))
    new_list += pad_dx
    return new_list

def align_tok_discard(list_1: list[str],
              list_2: list[str],
              discard: bool) -> list[str]:
    #l1 is always the shorter one
    #discard indicates if we want to discard from l2 or
    #if false if we want to pad l_1

    max_idx_start_most_sim = 0
    max_value_most_sim = -1
    for (idx_list_1, elem_1) in enumerate(list_1):
        limit_sx = idx_list_1
        num_elem_a_dx = len(list_1) - idx_list_1 -1
        limit_dx = len(list_2) - num_elem_a_dx
        l2_limited = list_2[limit_sx:limit_dx]
        # print(f"at idx_list_1: {idx_list_1}, limit_sx: {limit_sx}, limit_dx: {limit_dx}, elem_1: {elem_1}")
        # print(f"l2_limited: {l2_limited}")
        idx_most_sim_2, value_most_sim_2 = get_most_similar_token(elem_1, l2_limited )
        if value_most_sim_2 > max_value_most_sim:
            max_value_most_sim = value_most_sim_2
            max_idx_start_most_sim = idx_most_sim_2
    if discard:
        max_idx_end_most_sim = max_idx_start_most_sim + len(list_1)
        return list_2[max_idx_start_most_sim:max_idx_end_most_sim]
    else:
        padded_list = pad_list(list_1,
                               start_actual_list=max_idx_start_most_sim,
                               len_to_get=len(list_2))
        return padded_list


def align_tok(prediction_tok: list[str],
              tokenized_span: list[str]) -> list[str]:
    

    if len(prediction_tok) > len(tokenized_span):
        return align_tok_discard(list_1=tokenized_span,
                                 list_2=prediction_tok,
                                 discard=True)
    else:

        return align_tok_discard(list_1=prediction_tok,
                                 list_2=tokenized_span,
                                 discard=False)