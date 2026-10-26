import os
import argparse
from GFLDataLoader.GFLDataset import DatasetGFL, Sentence, Span
from generation.ModelRewriting import ModelRewritingDecoder
from itertools import product
from typing import Optional
from utils.files_io import read_jsonl

SYSTEM_PROMPT_DETECTION = """Sei un assistente di linguaggio italiano specializzato nel riconoscimento di espressioni non inclusive dal punto di vista del genere."""

SYSTEM_PROMPT_GENERATION = """Sei un assistente di linguaggio italiano specializzato nella riscrittura di espressioni non inclusive dal punto di vista del genere."""

TASKS = ["detection", "rewriting_free", "rewriting_context"]
#PREAMBLE_IDS = ["p1", "p2", "p3"]
PREAMBLE_IDS = ["p1"]
REWRITING_TYPES = ["io", "cv", "co", "iv"]

def get_all_combinations_prompts(tasks: list[str],
                                 rewriting_types: list[str]) -> list[dict]:
    # detection_combinations = list(product(["detection", "rewriting_free"], ["all"], PREAMBLE_IDS, [False, True], [SYSTEM_PROMPT_DETECTION]))
    # generation_combinations = list(product(["rewriting_context"], REWRITING_TYPES, PREAMBLE_IDS, [False, True], [SYSTEM_PROMPT_GENERATION]))
    #we only do fewshots
    detection_combinations = list(product(["detection", "rewriting_free"], ["all"], PREAMBLE_IDS,  [True], [SYSTEM_PROMPT_DETECTION]))
    generation_combinations = list(product(["rewriting_context"], rewriting_types, PREAMBLE_IDS, [True], [SYSTEM_PROMPT_GENERATION]))
    all_combinations = detection_combinations + generation_combinations

    all_combinations_dicts = [{
        "task": comb[0],
        "rewriting_type": comb[1],
        "preamble_id": comb[2],
        "fewshot": comb[3],
        "system_prompt": comb[4],
    } for comb in all_combinations if comb[0] in tasks]

    return all_combinations_dicts    

def get_detected_spans_from_bio(matching_sample: dict) -> list[str]:
    #from matching sample we look at the tokens in the field 'tokens' and we extract 
    #the spans. since each token has a BIO label in the field predicted label, we compose the string of the entire span.

    spans_sentence = []
    current_span = []
    for token in matching_sample["tokens"]:
        if token["predicted_label"].startswith("B-"):
            if current_span:
                spans_sentence.append(" ".join(current_span))
                current_span = []
            current_span.append(token["token"])
        elif token["predicted_label"].startswith("I-") and current_span:
            current_span.append(token["token"])
        else:
            if current_span:
                spans_sentence.append(" ".join(current_span))
                current_span = []
    if current_span:
        spans_sentence.append(" ".join(current_span))

    return spans_sentence

def align_matching_sample(matching_sample: dict,
                          sentence: Sentence) -> None:
    
    tokens_sample = matching_sample["tokens"]
    tokens_sentence = sentence.tokens
    if len(tokens_sample) > len(tokens_sentence):
        print(f"Found more tokens in sample than in sentence: {sentence.id}")
        print(f"Proceeding by truncating sample to sentence.tokens length")
        tokens_sample = tokens_sample[:len(tokens_sentence)]
    elif len(tokens_sample) < len(tokens_sentence):
        print(f"Found less tokens in sample than in sentence: {sentence.id}")
        print(f"Proceeding to add O predicted labels to all remaining tokens")
        for t_sentence in tokens_sentence[len(tokens_sample):]:
            #TODO: here we fix type_change to 'changed' but it has to be dynamically put based on the report
            new_token_sample = t_sentence.get_dict_with_label(type_change="changed")
            tokens_sample.append({
                "token": new_token_sample["token"],
                "predicted_label": "O",
                "true_label": new_token_sample["label"]
            })
    
    assert len(tokens_sample) == len(tokens_sentence), "After this preprocessing the two list of tokens should be the same len"

    for (t_sample, t_sentence) in zip(tokens_sample, tokens_sentence):
        if t_sample["token"] != t_sentence.token:
            print(f"Found misaligned token in sentence: {sentence.id}, t_sample: {t_sample['token']} and t_sentence: {t_sentence.token}")

def get_sample_detection(path_file_samples: str,
                       sentences: list[Sentence]) -> list[dict]:
    samples = read_jsonl(path_file_samples)
    # detected_spans_ordered = [next((d_s for d_s in detected_spans if d_s["sentence_id"] == s.id), [])
    #                           for s in sentences]

    samples_ordered = []
    for sentence in sentences:
        matching_sample = next((sample for sample in samples if sample["sentence_id"] == sentence.id), None)
        if matching_sample is None: 
            raise Exception(f"did not find a sample for sentence: {sentence.id}")
        
        align_matching_sample(matching_sample, sentence)
        detected_spans = get_detected_spans_from_bio(matching_sample)
        matching_sample["detected_spans"] = detected_spans
        samples_ordered.append(matching_sample)
    return samples_ordered

def get_spans_dataset(sentences: list[Sentence],
                      type_alternative: str) -> list[list[Span]]:

    #we get the spans to give to the model (we get the longest ones to avoid overlaps)
    if type_alternative=="all":
        print(f"When getting gold spans from dataset, found type_alternative: {type_alternative}, switching to None")
        type_alternative=None

    spans_dataset = []
    for sentence in sentences:
        spans_sentence = sentence.get_spans_to_sub(type_alternative=type_alternative)
        #spans_dataset.append([span.span for span in spans_sentence])
        #spans_dataset.append([span.get_dict() for span in spans_sentence])
        spans_dataset.append(spans_sentence)
    return spans_dataset

def main(out_dir: str,
         model_id: str,
         tokenizer_id: str,
         gpu_num: str,
         batch_size: int,
         tasks: list[str],
         rewriting_types: list[str],
         path_file_detected_spans: Optional[str] = None,
         **kwargs) -> None:

    num_sentences = kwargs.get("num_sentences", None)
    
    # prompt_maker = PromptMaker()
    # text_prompt = prompt_maker.get_prompt_text(task="rewriting", 
    #                                            rewriting_type="io", 
    #                                            sentence="Questo è un esempio di frase da riscrivere.", 
    #                                            preamble_id="p3", 
    #                                            detected_spans=["span1", "span2"], 
    #                                            fewshot=True)
    # print(text_prompt)
    # output = model_rewriting_decoder.generate([text_prompt])
    # print(output)

    prompt_args_combinations = get_all_combinations_prompts(tasks=tasks,
                                                            rewriting_types=rewriting_types)
    

    model_rewriting_decoder = ModelRewritingDecoder(model_id=model_id, 
                                                    tokenizer_id=tokenizer_id, 
                                                    gpu_num=gpu_num, 
                                                    out_dir=out_dir)
    gfl_dataset = DatasetGFL("./Data/gfl-it-new.json")

    #TODO: filter sentences (based if it is a rewriting task, if they have rewritings,...)
    sentences = [sent for ex in gfl_dataset.examples for sent in ex.sentences]
    if (num_sentences):
        sentences=sentences[:num_sentences]
    
    
    for prompt_args in prompt_args_combinations:
        print(f"-"*20)
        print(f"Task: {prompt_args['task']}, Rewriting type: {prompt_args['rewriting_type']}, Preamble: {prompt_args['preamble_id']}, Fewshot: {prompt_args['fewshot']}, Model: {model_id}")
        
        input_spans = None
        if prompt_args["task"] in ["rewriting_context", "rewriting_free"]:
            if path_file_detected_spans:
                print(f"Reading spans from detection file: {path_file_detected_spans}")
                sample_detection = get_sample_detection(path_file_detected_spans,
                                                    sentences)
                
                input_spans = [s["detected_spans"] for s in sample_detection]
                #TODO: evitare di fare generazioni se non ci sono detected_spans? Oppure affidarsi a modello e dirgli che se ne trova altre di cambiarle?
                #print(input_spans)
            else:
                print(f"Reading spans from dataset")
                input_spans = get_spans_dataset(sentences, prompt_args['rewriting_type'])
                #print("No path for detected spans provided, so no detected spans will be used in the prompts.")
                #print(input_spans)
        system_prompt = prompt_args.pop("system_prompt") #TODO: se facciamo loop su più modelli non va
        rewritings = model_rewriting_decoder.generate_from_sentences(
                                                        sentences=sentences,
                                                        prompt_args=prompt_args,
                                                        batch_size=batch_size,
                                                        system_prompt=system_prompt,
                                                        input_spans=input_spans
                                                        )
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--tokenizer_id", type=str, required=False)
    parser.add_argument("--gpu_num", type=str, required=False)
    parser.add_argument("--num_sentences", type=int, required=False, help="Parameter that can be set in order to run trials on just num_sentences")
    parser.add_argument("--batch_size", type=int, required=False, default=16, help="Batch size for generation")
    parser.add_argument("--path_file_detected_spans", type=str, required=False, help="Path to a file containing the detected spans for each sentence. If not provided, no detected spans will be used in the prompts.")
    parser.add_argument("--tasks", 
                        nargs="+",
                        choices=TASKS,
                        default=TASKS)
    parser.add_argument("--rewriting_types", 
                        nargs="+",
                        choices=REWRITING_TYPES,
                        default=REWRITING_TYPES)
    args = parser.parse_args()
    if args.tokenizer_id is None:
        if args.model_id is not None:
            args.tokenizer_id = args.model_id
        else:
            raise Exception("Could not set a tokenizer as both tokenizer_id and model_id are None")
    
    main(**vars(args))
