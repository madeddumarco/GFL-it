from typing import Union, Dict, Optional
from pathlib import Path
from utils.files_io import read_json, dump_json
from .Form import Form, FormChanged, FormMascFem, FormNeo, FormUnchanged
import json
import random
import itertools
from nltk import WordPunctTokenizer
from collections import defaultdict
from copy import deepcopy

class EncoderGFL(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "attr_to_save"):
            return {k:v for (k,v) in obj.__dict__.items() if k in obj.attr_to_save}
        
        #TODO: add other cases
        return json.JSONEncoder.default(self, obj)

class EncoderGFLEval(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Token):
            attr_token = ["token", "forms_cv", "forms_neo"]
            return {k:v for (k,v) in obj.__dict__.items() if k in attr_token}
        elif isinstance(obj, Form):
            
            return str(obj)
      
        #TODO: add other cases
        return json.JSONEncoder.default(self, obj)

class DatasetGFL():

    def __init__(self, 
                 dataset_path: Union[str, Path],
                 create_dict_token_forms: bool = False) -> None:
        dataset = read_json(dataset_path) #import in utils
        self.examples = [Example(**ex) for ex in dataset]
        self.attr_to_save = ["examples"]
        self.dict_token_forms = self._create_dict_token_forms() if create_dict_token_forms else None


    def _create_dict_token_forms(self) -> dict[str, set[Form]]:
        dict_t_f = defaultdict(set)
        #facciamo anche qua dict separati per cv, neom?
        for ex in self.examples:
            for sent in ex.sentences:
                for token in sent.tokens:
                    forms_all = token.forms_all
                    token_key = token.token_lower
                    dict_t_f[token_key].update(forms_all)

        return dict_t_f
    
    def save_to_json(self, file_name: str) -> None:
        dump_json(self.examples, file_name, cls=EncoderGFL)            

    def get_example_by_id(self, id: str, throw_error: bool = False) -> Union["Example", None]:
        ex_found = next((ex for ex in self.examples if ex.id == id), None)
       
        if throw_error and (ex_found is None):
            raise Exception(f"example with id: {id} not found")
        return ex_found
    
    def get_sentence_by_id(self, id: str, throw_error: bool = False) -> Union["Sentence", None]:
        sentences = [sent for ex in self.examples for sent in ex.sentences]
        sent_found = next((sent for sent in sentences if sent.id == id), None)
       
        if throw_error and (sent_found is None):
            raise Exception(f"sentence with id: {id} not found")
        return sent_found

    def get_tokens_with_label(self, get_bio: bool = False, type_change: str = "changed") -> list[list[dict]]:
        """returns a list of sentences divided in tokens with labels for unfair-detection, each element in the list is a list of dicts with token and label
         type_change: "changed" (which returns tokens that have changed in the rewriting for IO or CV) or "is_in_span" (which returns tokens that are inside the annotated unfair spans)
         get_bio: if True, labels are in BIO format
        
        """
        tokens_with_label = {sent.id: sent.get_tokens_with_label(type_change = type_change) for ex in self.examples for sent in ex.sentences} 
        if get_bio:
            return self.transform_list_to_BIO(tokens_with_label, return_type="dict")
        return tokens_with_label
    
    def get_format_for_tokenizer(self, get_bio: bool = False, type_change: str = "changed") -> list[list[dict]]:
        """
       Returns a list of dictionaries. Each dictionary represents once sentence and has 3 keys:
       - "id": the id of the sentence
       - "tokens": list of all tokens in the sentence
       - "labels": list of the corresponding labels per token

       This is necessary to provide the supported format for the tokenizer
        type_change: "changed" (which returns tokens that have changed in the rewriting for IO or CV) or "is_in_span" (which returns tokens that are inside the annotated unfair spans)
        get_bio: if True, labels are in BIO format
        """
        tokens_with_label = self.get_tokens_with_label(get_bio=get_bio, type_change=type_change)

        format_dataset = []
        for (id_sentence, sent_tokens) in tokens_with_label.items():
            format_sent_tokens = {"id_sentence": id_sentence, "tokens": [], "labels": []}
            for token in sent_tokens:
                format_sent_tokens["tokens"].append(token["token"])
                format_sent_tokens["labels"].append(str(token["label"])) #str conversion necessary for evaluate
            format_dataset.append(format_sent_tokens)
            
        return format_dataset
    
    def get_sentences_texts(self) -> list[str]:
        """returns a list of all sentences texts in the dataset"""
        return [sent.sentence for ex in self.examples for sent in ex.sentences]
    
    @staticmethod
    def transform_list_to_BIO(tokens_with_label: Union[Dict[str, list[dict]], list[list[dict]]],
                              return_type: str) -> Union[Dict[str, list[dict]], list[list[dict]]]:
        """transforms a list of sentences with tokens and binary labels to BIO format"""
        assert return_type in ["list", "dict"], f"return_type must be one of [dict, list], got {return_type} instead"
        
        all_sentences_BIO = defaultdict(list)

        if isinstance(tokens_with_label, list):
            #if it's a list we assume it's a list of sentences with tokens and labels, so we create a dict with sentence id as key and list of tokens with label as value
            tokens_with_label = {f"sentence_{i}": sent for (i, sent) in enumerate(tokens_with_label)}
            
        for (sentence_id, sentence_tokens) in tokens_with_label.items():
            prev_label = 0
            for token_dict in sentence_tokens:
                token_label = token_dict["label"]
                if token_label == 1:
                    if prev_label == 0:
                        bio_label = "B-UNFAIR"
                    else:
                        bio_label = "I-UNFAIR"
                else:
                    bio_label = "O"
                token_dict_BIO = {
                    "token": token_dict["token"],
                    "label": bio_label
                }
                all_sentences_BIO[sentence_id].append(token_dict_BIO)
                prev_label = token_label

        if return_type == "list":
            return list(all_sentences_BIO.values())
        elif return_type == "dict":
            return all_sentences_BIO
    
    def create_sentences_preselect(self, num_new: int, type_alternative: str, type_change: str, return_type: str = "sentence") -> list[dict]: #the return type has to be decided, do we return the same things as get_tokens_with_label or is it optional?
        """returns a list of sentences created by substituting the spans with their alternatives, the number of sentences created is defined by num_new, the type of alternative is defined by type_alternative (IO or CV), the type of label for unfair-detection is defined by type_change (changed or is_in_span), the return type can be either "sentence" (which returns the new sentence text) or "list" (which returns a list of dicts with token and label for unfair-detection)"""
        return [sent.create_sentences_preselect(num_new=num_new, type_alternative=type_alternative, type_change=type_change, return_type=return_type) for ex in self.examples for sent in ex.sentences]
    
    
class Example():

    def __init__(self, 
                 id: str, 
                 text: str, 
                 sentences: list[dict],
                 **kwargs) -> None:
        
        self.id = id
        self.text = text
        self.sentences = [Sentence(**sent) for sent in sentences]
        self.attr_to_save = ["id", "text", "sentences"]

    def get_sentence_by_id(self, id: str, throw_error: bool = False) -> Union["Sentence", None]:
        sent_found = next((ex for ex in self.sentences if ex.id == id), None)
        if throw_error and (sent_found is None):
            raise Exception(f"sentence with id: {id} not found")
        return sent_found

    def get_tokens_with_label(self, type_change: str = "changed") -> list[list[dict]]:
        """returns a list of sentences divided in tokens with labels for unfair-detection, each element in the list is a list of dicts with token and label
         type_change: "changed" (which returns tokens that have changed in the rewriting for IO or CV) or "is_in_span" (which returns tokens that are inside the annotated unfair spans)
        
        """
        return [sent.get_tokens_with_label(type_change = type_change) for sent in self.sentences]

    def create_sentences_preselect(self, num_new: int, type_alternative: str, type_change: str, return_type: str = "sentence") -> list[dict]: #the return type has to be decided, do we return the same things as get_tokens_with_label or is it optional?
        """returns a list of sentences created by substituting the spans with their alternatives, the number of sentences created is defined by num_new, the type of alternative is defined by type_alternative (IO or CV), the type of label for unfair-detection is defined by type_change (changed or is_in_span), the return type can be either "sentence" (which returns the new sentence text) or "list" (which returns a list of dicts with token and label for unfair-detection)"""
        return [sent.create_sentences_preselect(num_new=num_new, type_alternative=type_alternative, type_change=type_change, return_type=return_type) for sent in self.sentences]
class Sentence():

    def __init__(self, 
                 id: str, 
                 sentence: str, 
                 start_idx_text: int,
                 end_idx_text: int,
                 label: str,
                 spans: list[dict],
                 tokens: list[dict],
                 **kwargs) -> None:

        #other parameters
        self.id = id
        self.idx = int(self.id.split("_")[1])
        self.sentence = sentence
        self.start_idx_text = start_idx_text
        self.end_idx_text = end_idx_text
        self.label = label
        self.spans = [Span(sentence_id=self.id, **span) for span in spans]
        self.tokens = [Token(sentence_base=start_idx_text, **token) for token in tokens]
        self.attr_to_save = ["id", "sentence", "start_idx_text", "end_idx_text", "label", "spans", "tokens"]

    def get_dict(self) -> dict:
        fields_to_return = ["id", "sentence", "start_idx_text", "end_idx_text"]
        return {k:v for (k,v) in self.__dict__.items() if k in fields_to_return}
    
    def print_tokens(self) -> None:
        for t in self.tokens:
            print(t)

    def get_span_by_id(self, id_span: str, throw_error: bool = False) -> Union["Span", None]:
        span_found = next((span for span in self.spans if span.id == id_span), None)
        if throw_error and span_found is None:
            raise Exception(f"span with id: {id} not found")
        return span_found

    def get_tokens_with_label(self, type_change: str = "changed") -> list[dict]:
        """returns a list of tokens with labels for unfair-detection, each element in the list is a list of dicts with token and label
         type_change: "changed" (which returns tokens that have changed in the rewriting for IO or CV) or "is_in_span" (which returns tokens that are inside the annotated unfair spans)
        
        """
        return [token.get_dict_with_label(type_change = type_change) for token in self.tokens]
    
    def get_spans_to_sub(self, type_alternative: str= None,
                         fixed_spans: dict = None) -> list["Span"]:
        """
        Returns a list of non-overlapping spans, greedily selecting 
        the longest span when conflicts arise.
        If type_alternative is given, each span has at least a rewriting for said alternative. If type_alternative is None, all spans are selected no matter rewritings.
        """

          
        # Sort by length descending (longest first), then by start index
        sorted_spans = sorted(
            self.spans,
            key=lambda s: (-(s.end_idx_sentence - s.start_idx_sentence), s.start_idx_sentence)
        )

        if fixed_spans is not None:
            id_spans_fixed = fixed_spans.keys()
            try:
                spans_fixed = [self.get_span_by_id(id_span) for id_span in id_spans_fixed]
                selected = spans_fixed
            except Exception as e:
                raise Exception(f"When trying to get the spans fixed for substitution we got error: {e}")
        else:
            selected = []
            
        for candidate in sorted_spans:
            # Check if candidate overlaps with any already-selected span
            if (((type_alternative is not None) 
                    and len(candidate.get_rewritings(type_alternative))) 
                or (type_alternative is None)):

                overlaps = any(
                    candidate.start_idx_sentence < chosen.end_idx_sentence and
                    candidate.end_idx_sentence > chosen.start_idx_sentence
                    for chosen in selected
                )
                if not overlaps:
                    selected.append(candidate)
            

        # Return in order of appearance in the sentence
        return sorted(selected, key=lambda s: s.start_idx_sentence)
    
    def get_tokens_range(self, start_idx: int, end_idx: int, type_change: str) -> list["Token"]:
        tokens_in_range = []
        i_token = 0
        while (i_token < len(self.tokens) and self.tokens[i_token].start_idx_sentence < end_idx):
            if self.tokens[i_token].end_idx_sentence > start_idx:
                tokens_in_range.append(self.tokens[i_token].get_dict_with_label(type_change=type_change))
            i_token += 1
        
        return tokens_in_range
        
    
    def sub_spans(self, spans_to_sub: list["Span"], new_sentence: str, new_tokens: list, type_alternative: str, 
                  type_change: str, end_previous_span: int, idx_span: int, return_type: str, chosen_rewritings: list,
                  transform_bio: bool = None, tokenizer = None) -> list:
        if idx_span>= len(spans_to_sub):
            new_tokens.extend(self.get_tokens_range(start_idx=end_previous_span, end_idx=len(self.sentence), type_change=type_change))
            new_sentence += self.sentence[end_previous_span:]

            if return_type == "sentence":
                return new_sentence
            elif return_type == "list":
                if transform_bio:
                    return DatasetGFL.transform_list_to_BIO([new_tokens], return_type="list")[0]
                else:
                    return new_tokens
            elif return_type == "both":
                if transform_bio:
                    new_tokens = DatasetGFL.transform_list_to_BIO([new_tokens], return_type="list")[0]
                return {"sentence": new_sentence, "tokens": new_tokens}
            else:
                raise Exception("return type not in allowed choices: sentence, list")
        
        span_to_sub = spans_to_sub[idx_span]
        start_span_to_sub = span_to_sub.start_idx_sentence
        end_span_to_sub = span_to_sub.end_idx_sentence

        #first we add the tokens that are before the new span 
        new_tokens.extend(self.get_tokens_range(start_idx=end_previous_span, end_idx=start_span_to_sub, type_change=type_change))
        new_sentence += self.sentence[end_previous_span:start_span_to_sub]

        rew_to_add = chosen_rewritings[idx_span]
        new_sentence += rew_to_add
        tokenized_rew = tokenizer.tokenize(rew_to_add)
        tokenized_rew_dicts = [{    
                                    "token": token,
                                    "label": 0 #label is set to 0 as a standard because we assume the rewriting is not an unfair case, this is a simplification 
                                } for token in tokenized_rew]

        new_tokens.extend(tokenized_rew_dicts)
        
        return self.sub_spans(spans_to_sub=spans_to_sub,
                       new_sentence=new_sentence,
                       new_tokens=new_tokens,
                       type_alternative=type_alternative,
                       type_change=type_change,
                       end_previous_span=end_span_to_sub,
                       idx_span= idx_span+1,
                       return_type=return_type,
                       chosen_rewritings=chosen_rewritings,
                       transform_bio=transform_bio, 
                       tokenizer=tokenizer)

    def _get_possible_rews(self,
                      spans_to_sub: list["Span"],
                      type_alternative: str,
                      fixed_spans: dict = None) -> list[list[str]]:
        
        possible_rews = []
        for span in spans_to_sub:
            if ((fixed_spans is not None) and
                 ((span.id in fixed_spans)) and fixed_spans[span.id] != ""):
                possible_rews.append([fixed_spans[span.id]])
            else:
                possible_rews.append(span.get_rewritings(type_alternative))
        return possible_rews

    def create_sentences_preselect(self, 
                                   num_new: int, 
                                   type_alternative: str, 
                                   type_change: str = "changed", 
                                   return_type: str = "sentence", 
                                   transform_bio: bool = None,
                                   fixed_spans: dict = None) -> list[dict]: #the return type has to be decided, do we return the same things as get_tokens_with_label or is it optional?
        """returns a list of sentences created by substituting the spans with their alternatives, the number of sentences created is defined by num_new, the type of alternative is defined by type_alternative (IO or CV), the type of label for unfair-detection is defined by type_change (changed or is_in_span), the return type can be either "sentence" (which returns the new sentence text) or "list" (which returns a list of dicts with token and label for unfair-detection)
         num_new: number of new sentences to create
         type_alternative: type of alternative to use for substitution (e.g. IO or CV)
         type_change: type of label for unfair-detection (changed or is_in_span) -> deve essere opzionale
         return_type: "sentence" (which returns the new sentence text) or "list" (which returns a list of dicts with token and label for unfair-detection)
        """
        assert return_type in ["sentence", "list", "both"], f"return_type: {return_type} not in allowed choices: sentence, list"
        token_lists_substitutions = []
        spans_to_sub = self.get_spans_to_sub(type_alternative=type_alternative,
                                             fixed_spans=fixed_spans)
        #possible_rews = [s.get_rewritings(type_alternative) for s in spans_to_sub]
        possible_rews = self._get_possible_rews(spans_to_sub=spans_to_sub,
                                                type_alternative=type_alternative,
                                                fixed_spans=fixed_spans)
        #print(f"possible_rews: {possible_rews}")
        output_list = list(itertools.product(*possible_rews))
        num_new = min(num_new, len(output_list))
        chosen_rewritings = random.sample(output_list, num_new)
        #print(f"chosen_rewritings: {chosen_rewritings}")
        
        tokenizer = WordPunctTokenizer()

        for i in range(num_new):
            new_sentence = ""
            new_tokens = []
            new_token_list = self.sub_spans(spans_to_sub = spans_to_sub, new_sentence = new_sentence, new_tokens = new_tokens, 
                                            type_alternative = type_alternative, type_change = type_change, end_previous_span=0, 
                                            idx_span=0, return_type=return_type, chosen_rewritings=chosen_rewritings[i], transform_bio=transform_bio,
                                            tokenizer=tokenizer)
            token_lists_substitutions.append(new_token_list)
        return token_lists_substitutions
    
    def get_tokens_in_span(self, span: "Span") -> list["Token"]:
        """returns a list of tokens that are inside the span"""
        return [token for token in self.tokens if token.start_idx_sentence >= span.start_idx_sentence and token.end_idx_sentence <= span.end_idx_sentence]
        #return [token for token in self.tokens if (token.start_idx_sentence >= span.start_idx_sentence and token.start_idx_sentence < span.end_idx_sentence) or (token.end_idx_sentence > span.start_idx_sentence and token.end_idx_sentence <= span.end_idx_sentence)]

class Span():

    def __init__(self, 
                 id: str,
                 sentence_id: str,
                 span: str, 
                 start_idx_text: int,
                 end_idx_text: int,
                 start_idx_sentence: int,
                 end_idx_sentence: int,
                 alternatives: dict,
                 **kwargs
                 ) -> None:

        self.id = id[0] if isinstance(id, list) else id
        self.sentence_id = sentence_id #inherited by sentence, not saved in the json
        self.span = span
        self.start_idx_text = start_idx_text
        self.end_idx_text = end_idx_text
        self.start_idx_sentence = start_idx_sentence
        self.end_idx_sentence = end_idx_sentence
        self.alternatives = alternatives
        self.attr_to_save = ["id", "span", "start_idx_text", "end_idx_text", 
                             "start_idx_sentence", "end_idx_sentence", 
                             "alternatives"]

    def __str__(self):
        return f"""span: {self.span},
        start_idx_text: {self.start_idx_text},
        end_idx_text: {self.end_idx_text},
        start_idx_sentence: {self.start_idx_sentence},
        end_idx_sentence: {self.end_idx_sentence}"""

    def get_dict(self) -> dict:
        fields_to_return = ["id", "span",
                            "start_idx_sentence", "end_idx_sentence"]
        return {k:v for (k,v) in self.__dict__.items() if k in fields_to_return}
    
    def refresh_id(self) -> None:
        """refreshes the id of the span based on its content and position in the sentence"""
        self.id = "_".join([self.span, str(self.start_idx_sentence), str(self.end_idx_sentence)])
    
    def get_random_rewriting(self, type_rewriting: str):
        """returns a random rewriting for the span based on the type of rewriting (IO or CV)"""
        if type_rewriting not in self.alternatives:
            raise Exception(f"type_rewriting: {type_rewriting} not found in alternatives")
        if len(self.alternatives[type_rewriting]) == 0:
            raise Exception(f"no alternatives found for type_rewriting: {type_rewriting}")
        return random.choice(self.alternatives[type_rewriting])

    def get_rewritings(self, type_rewriting: str) -> list:
        type_rewriting=type_rewriting.upper()
        if type_rewriting not in self.alternatives:
            raise Exception(f"type_rewriting: {type_rewriting} not found in alternatives")

        return self.alternatives[type_rewriting]
    

class Token():

    def __init__(self, 
                 token: str,
                 token_lower: str,
                 start_idx_sentence: int,
                 end_idx_sentence: int, 
                 forms_neo: list[dict],
                 forms_cv: list[dict],
                 sentence_base: int, 
                 is_in_span: bool,
                 **kwargs
                 ) -> None:

        self.token = token
        self.token_lower = token_lower
        self.start_idx_sentence = start_idx_sentence
        self.end_idx_sentence = end_idx_sentence
        self.start_idx_text = sentence_base + start_idx_sentence
        self.end_idx_text = sentence_base + end_idx_sentence
        self.is_in_span = is_in_span
        self.forms_cv = [Form.from_dict(form_cv) for form_cv in forms_cv]
        self.forms_neo = [Form.from_dict(form_neo) for form_neo in forms_neo]
        self.attr_to_save = ["token", "token_lower", "start_idx_text", "end_idx_text", 
                             "start_idx_sentence", "end_idx_sentence", 
                             "forms_neo", "forms_cv", "is_in_span"]
        self.forms_all = self.get_all_forms()
        
    def get_all_forms(self) -> set["Form"]:
        return set(self.forms_cv + self.forms_neo)

    def __str__(self):
        return f"""token: {self.token},
        start_idx_sentence: {self.start_idx_sentence},
        end_idx_sentence: {self.end_idx_sentence},
        start_idx_text: {self.start_idx_text},
        end_idx_text: {self.end_idx_text}"""
    
    def get_label(self, type_change: str = "changed") -> int:
        """returns the label for unfair-detection of the token
         type_change: "changed" (which returns tokens that have changed in the rewriting for IO or CV) or "is_in_span" (which returns tokens that are inside the annotated unfair spans)
            
        """
        if type_change == "changed":
            return int(any(form.is_changed() for form in self.forms_all))
        elif type_change == "is_in_span":
            return int(self.is_in_span)
        else:
            raise Exception(f"type_change: {type_change} not recognized")
        
    def get_dict_with_label(self, type_change: str = "changed") -> dict:
        """returns a dict with token and label for unfair-detection
         type_change: "changed" (which returns tokens that have changed in the rewriting for IO or CV) or "is_in_span" (which returns tokens that are inside the annotated unfair spans)
        
        """
        return {
            "token": self.token,
            "label": self.get_label(type_change=type_change)
        }
    
    def _get_cv_forms_equal(self,
                        token_1: str,
                        token_2: str,
                        list_forms: set[Form] = None) -> list['Form']:
        
        forms_cv = [f for f in list_forms if isinstance(f, FormMascFem)]
        form_cv_equal_1_2 = [f for f in forms_cv if f.is_same_rew((token_1, token_2))]
        form_cv_equal_2_1 = [f for f in forms_cv if f.is_same_rew((token_2, token_1))]
        return list(set(form_cv_equal_1_2 + form_cv_equal_2_1))
    
    def _get_neo_forms_equal(self,
                        token_1: str,
                        list_forms: set[Form]) -> list['Form']: 
        
        form_neom = [f for f in list_forms if isinstance(f, FormNeo)]
        form_neom_equal = [f for f in form_neom if f.is_same_rew(token_1)]
        return form_neom_equal

    def is_valid_rewriting_io(self,
                              rewriting: str,
                              list_forms: set[Form] = None,
                              return_type_error: bool = False) -> Union[str, tuple]:
    
        result = ""
        type_errors = []
        if "<neom>" in rewriting:
            
            form_neom_equal = self._get_neo_forms_equal(token_1=rewriting, list_forms=list_forms)
            result = "true_pos" if (len(form_neom_equal)>0) else "false_pos"
            if result == "false_pos":
                type_error = "unnecessary_neom" if len([f for f in list_forms if isinstance(f, FormNeo)]) == 0 else "wrong_neom"
                type_errors.append(type_error)
        else:
            form_others = [f for f in list_forms 
                           if isinstance(f, FormChanged) or isinstance(f, FormUnchanged)]
            form_others_equal = [f for f in form_others if f.is_same_rew(rewriting)]
            result = "true_neg" if (len(form_others_equal)>0) else "false_neg"
            if result == "false_neg":
                type_error = "missed_neom" if len([f for f in list_forms if isinstance(f, FormNeo)]) > 0 else "wrong_token"
                type_errors.append(type_error)

        if len(type_errors) == 0:
            type_errors.append("correct")
        if return_type_error:
            return (result, type_errors)
        else:
            return result
    
    def is_valid_rewriting_cv(self,
                              rewriting: str,
                              list_forms: set[Form] = None,
                              return_type_error: bool = False) -> Union[str, tuple]:
    
        result = ""
        type_errors = []
        if "/" in rewriting:
            #in this case the model generated the rewriting for gender
            split_rewriting = rewriting.split("/")
            if len(split_rewriting) == 2:
                #this is the correct case for a cv rewriting (two forms)
                masc_form, fem_form = split_rewriting #this is because we instructed models to first generate the male form and then the female one
                form_cv_equal = self._get_cv_forms_equal(token_1=masc_form, token_2=fem_form, list_forms=list_forms)
                result = "true_pos" if (len(form_cv_equal)>0) else "false_pos"
                if result == "false_pos":
                    type_error = "unnecessary_mf" if len([f for f in list_forms if isinstance(f, FormMascFem)]) == 0 else "wrong_mf"
                    type_errors.append(type_error)
            else:

                result = "false_pos"
                type_error = "wrong_number_slash"
                type_errors.append(type_error)
        else:
            #qua gli unchanged secondo me devono essere più strict che neom -> se esiste forma femminile va messa?
            
            # forms_cv = [f for f in list_forms if isinstance(f, FormMascFem)]
            # if len(forms_cv)>0:
            #     result = "false_neg"
            # else:
                form_others = [f for f in list_forms
                           if isinstance(f, FormChanged) or isinstance(f, FormUnchanged)]
                form_others_equal = [f for f in form_others if f.is_same_rew(rewriting)]
                result = "true_neg" if (len(form_others_equal)>0) else "false_neg"
                if result == "false_neg":
                    type_error = "missed_mf" if len([f for f in list_forms if isinstance(f, FormMascFem)]) > 0 else "wrong_token"
                    type_errors.append(type_error)

        if len(type_errors) == 0:
                    type_errors.append("correct")
        if return_type_error:
            return (result, type_errors)
        else:
            return result
    
    def is_valid_rewriting_iv(self,
                              rewriting: str,
                              list_forms: set[Form] = None,
                              return_type_error: bool = False) -> Union[str, tuple]:
        
        def print_info_forms(list_form: list["Form"],
                             name_list: str) -> None:
            print(f"len({name_list}): {len(list_form)}")
            print(f"start list: {name_list}")
            for f in list_form:
                print(f)
            print(f"end list: {name_list}")

        def _get_m_f_n(split_rewriting: list[str]) -> tuple[str, str, str]:
            #assumes that split_rewriting has len = 3.
            split_rewriting = deepcopy(split_rewriting)
            idx_neom = next((i for (i, elem) in enumerate(split_rewriting) if "<neom>" in elem), 2)
            form_neom = split_rewriting.pop(idx_neom)
            form_masc, form_fem = split_rewriting
            return form_masc, form_fem, form_neom

        result = ""
        type_errors = []
        # print(f"token: {self.token}, rewriting: {rewriting}")
        if "/" in rewriting:
            #in this case the model generated the rewriting for gender
            split_rewriting = rewriting.split("/")
            # print("--------------------------------------")
            # print(f"split_rewriting: {split_rewriting}")
            if len(split_rewriting) == 3:
                #this is the correct case for a cv rewriting (two forms)
                masc_form, fem_form, neo_form = _get_m_f_n(split_rewriting)
                
                form_cv_equal = self._get_cv_forms_equal(token_1=masc_form,
                                                              token_2=fem_form,
                                                              list_forms=list_forms)
                
                form_neom_equal = self._get_neo_forms_equal(token_1=neo_form, list_forms=list_forms)
                
                # print_info_forms(forms_cv, "forms_cv")
                # print_info_forms(form_cv_equal, "forms_cv_equal")
                # print_info_forms(form_neom, "form_neom")
                # print_info_forms(form_neom_equal, "form_neom_equal")
                # print(f"masc_form: {masc_form}, fem_form:{fem_form}, neo_form: {neo_form}")
                # print(f"len(form_cv_equal)>0{(len(form_cv_equal)>0)}, len: {len(form_cv_equal)}")
                # print(f"len(form_neom_equal)>0{(len(form_neom_equal)>0)}, len: {len(form_neom_equal)}")
                # print(f"{(len(form_cv_equal)>0) and len(form_neom_equal)>0}")
                result = "true_pos" if ((len(form_cv_equal)>0) and len(form_neom_equal)>0) else "false_pos"
                # print(f"result: {result}")
                if result == "false_pos":
                    type_error_m_f = "unnecessary_mf" if len([f for f in list_forms if isinstance(f, FormMascFem)]) == 0 else "wrong_mf"
                    type_error_neom = "unnecessary_neom" if len([f for f in list_forms if isinstance(f, FormNeo)]) == 0 else "wrong_neom"
                    type_errors.append(type_error_m_f)
                    type_errors.append(type_error_neom)

            else:
                # print(f"len not 3")
                result = "false_pos" #TODO: vedere se questo è da considerare false pos o false neg (anche per cv)
                
                type_errors.append("wrong_number_slash")
        else:
            #qua gli unchanged secondo me devono essere più strict che neom -> se esiste forma femminile va messa?
            
            #forms_cv = [f for f in list_forms if isinstance(f, FormMascFem)]
            #if len(forms_cv)>0:
            #    result = "false_neg"
            #else:
                form_others = [f for f in list_forms
                           if isinstance(f, FormChanged) or isinstance(f, FormUnchanged)]
                form_others_equal = [f for f in form_others if f.is_same_rew(rewriting)]
                result = "true_neg" if (len(form_others_equal)>0) else "false_neg"
                if result == "false_neg":
                    type_error_m_f = "missed_mf" if len([f for f in list_forms if isinstance(f, FormMascFem)]) > 0 else "wrong_token"
                    type_error_neom = "missed_neom" if len([f for f in list_forms if isinstance(f, FormNeo)]) > 0 else "wrong_token"
                    type_errors.append(type_error_m_f)
                    type_errors.append(type_error_neom)

        if len(type_errors) == 0:
            type_errors.append("correct")
                    
        if return_type_error:
            return (result, type_errors)
        else:
            return result

    def is_valid_rewriting(self, 
                           rewriting: str,
                           type_rewriting: str,
                           possible_forms: Optional[set[Form]] = None,
                            return_type_error: bool = False) -> Union[str, tuple]:
        
        type_rewriting = type_rewriting.upper()
        rewriting = rewriting.casefold()
        list_forms = possible_forms if possible_forms is not None else self.forms_all
        if type_rewriting == "IO":
            return self.is_valid_rewriting_io(rewriting, list_forms=list_forms, return_type_error=return_type_error)
        elif type_rewriting == "CV":
            return self.is_valid_rewriting_cv(rewriting, list_forms=list_forms, return_type_error=return_type_error)
        elif type_rewriting == "IV":
            return self.is_valid_rewriting_iv(rewriting, list_forms=list_forms, return_type_error=return_type_error)
        else:
            raise NotImplementedError(f"Function for checking is valid rewriting for {type_rewriting} has not been implemented yet")




    