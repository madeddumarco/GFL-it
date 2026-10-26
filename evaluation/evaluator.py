from GFLDataLoader.GFLDataset import DatasetGFL, Sentence, Span, Token, EncoderGFLEval
import abc
from typing import Union, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report
import torch
import re
from utils.files_io import dump_json, dump_jsonl, read_jsonl, join_paths, create_dir
from pathlib import Path
from .utils_eval import nltk_tokenize, merge_slash_tokens, get_csv_tokens, split_list_for_csv, interleave_indexed_dicts, write_csv_samples, align_tok
from copy import deepcopy
from collections import defaultdict

class Evaluator(abc.ABC):
    
    def __init__(self,
                 task_name: str,
                 subtask_name: str,
                 model_id: str,
                 out_dir: Optional[Path] = None,
                 dataset: Optional[DatasetGFL] = None,
                 save_samples: bool = False,
                 use_dict_forms: bool = False
                 ) -> None:
        super().__init__()
        self.task_name = task_name
        self.subtask_name = subtask_name
        self.model_id = model_id
        self.out_dir = out_dir
        self.use_dict_forms = use_dict_forms
        self.dataset = dataset if dataset is not None else DatasetGFL("Data/gfl-it-new.json", create_dict_token_forms=self.use_dict_forms)
        self.save_samples = save_samples
        self.report = {}
        self.samples = None

    @abc.abstractmethod
    def _get_data_csv(self, data_jsonl: list[dict]) -> Union[list[dict], None]:
        pass

    def _dump_report(self) -> None:
        model_id_safe = self.model_id.replace("/", " ")
        dir_path_report = join_paths(self.out_dir, self.task_name, self.subtask_name, model_id_safe )
        create_dir(dir_path_report)
        file_path_report = join_paths(dir_path_report, "report.json")
        dump_json(self.report, file_path_report)
        if self.save_samples and self.samples is not None:
            file_path_samples = join_paths(dir_path_report, "samples.jsonl")
            dump_jsonl(self.samples, file_path_samples, encoder=EncoderGFLEval)
            data_csv = self._get_data_csv(self.samples)
            if data_csv is not None:
                file_path_csv = join_paths(dir_path_report, "samples.csv")
                write_csv_samples(data_csv, file_path_csv)

    def _check_pred_ref(self,
                       predictions: list,
                       references: list) -> None:
        
        if len(predictions) != len(references):
            raise Exception(f"For {self.task_name} we got a different number of predictions ({len(predictions)}) and references ({len(references)})")
        
        if len(predictions) == 0:
            raise Exception(f"For {self.task_name} we got 0 predictions ({len(predictions)})")


    def _preprocess_output(self, output: str) -> str:
        """Only take the first line of the output, and remove known prefixes."""
        lines = output.splitlines()
        text = lines[0] if lines else output
        text = text.strip()

        for prefix in ("[Riscrittura]:", "[Span]:"):
            text = text.removeprefix(prefix).strip()

        return text
    def _get_sentences_sp_tokens_pred(self, samples: list[dict]) -> tuple[list[Sentence], list[Span], list[list[Token]], list[str]]:
        #divide samples in outputs and sentences and span
        sentences = []
        spans = []
        spans_tokens = []
        predictions = []
        
        for sample in samples:
            sentence = self.dataset.get_sentence_by_id(sample["id_sentence"])
            span = sentence.get_span_by_id(sample["id_span"])
            sample["processed_output"] = self._preprocess_output(sample["output"])
            prediction = sample["output"]
            span_tokens = sentence.get_tokens_in_span(span)

            sentences.append(sentence)
            spans.append(span)
            spans_tokens.append(span_tokens)
            predictions.append(prediction)
        return sentences, spans, spans_tokens, predictions
    @abc.abstractmethod
    def evaluate(self,
                 samples: list,
                 **kwargs) -> Union[float|dict]: #TODO: forse anche altro oltre a float
        pass


class EvaluatorDetection(Evaluator):

    
    def __init__(self,
                 type_change: str,
                 bio: bool,
            **kwargs,) -> None:
        """type_change indicates how we want to check the tokens: if they part of span ('is_in_span') or if they changed in rewriting ('changed')
        bio: indicates if evaluation is done with BIO or not"""
        super().__init__(**kwargs)
        self.type_change = type_change
        #TODO: actually implement BIO
        self.bio = bio

    
    def _get_data_csv(self, data_jsonl: list[dict]) -> Union[list[dict], None]:
        return None
    
    #TODO: check how well the boundaries work: see if things do not get found or other
    def _find_substring_intervals(self, text: str, substrings: list[str]) -> list[tuple[int, int]]:
        """
        Return a list of (start, end) inclusive intervals where one of the
        substrings occurs as a whole-word (respecting word boundaries).

        Parameters
        ----------
        text : str
        substrings : list[str]
            Ordered from longest to shortest (ordering is not required).

        Returns
        -------
        list[tuple[int, int]]
        """
        if not substrings:
            return []

        # Escape special regex characters
        escaped = [re.escape(s) for s in substrings]

        # Match any substring, respecting word boundaries
        pattern = r"\b(?:%s)\b" % "|".join(escaped)

        return [(m.start(), m.end()) for m in re.finditer(pattern, text)]

    def _map_pred_to_tokens(self,
                            prediction: list[str],
                            sentence: Sentence) -> list[dict]:
        def _is_in_interval(idx_start: int,
                            idx_end: int,
                            intervals: list[tuple[int, int]]) -> bool:
            for interval in intervals:
                if idx_start >= interval[0] and idx_end <= interval[1]:
                    return True
            return False

        sentence_text = sentence.sentence.lower()
        intervals_pred = self._find_substring_intervals(sentence_text, prediction)
        pred_tokens_with_label = []

        #TODO: fix the labels 0 and 1 in case of BIO
        for token in sentence.tokens:
            dict_token = {"token": token.token}
            if _is_in_interval(token.start_idx_sentence, token.end_idx_sentence, intervals_pred):
                dict_token["label"] = 1
            else:
                dict_token["label"] = 0
            pred_tokens_with_label.append(dict_token)
        return pred_tokens_with_label

    def _preprocess_predictions(self, predictions: list) -> list[list[str]]:
            pred0 = predictions[0]
            if isinstance(pred0, str):
                predictions = [p.split(";") for p in predictions]
            elif not isinstance(pred0, list):
                raise Exception(f"We expected predictions to be either an unprocessed str or a list but we got a list of: {type(pred0)}")
            
            predictions_processed = []
            for pred in predictions:
                pred_lower = [p.lower().strip() for p in pred]
                pred_set = set(pred_lower) #we avoid duplicates as there is no real mapping done by llms 
                pred_valid = [p for p in pred_set if p!=""]
                #we sort by len so that when mapping it to tokens it becomes easier with overlaps
                pred_sorted = sorted(pred_valid, key=lambda x: len(x), reverse=True) 
                predictions_processed.append(pred_sorted)

            return predictions_processed
    
    #TODO: da controllare
    def _compute_classification_report(self, predictions: list[dict], gold_labels: list[dict]) -> dict:
        """Generate a scikit-learn classification report from token-level predictions."""
        flattened_preds = [pred["label"] for pred in predictions]
        flattened_golds = [gold["label"] for gold in gold_labels]

        return classification_report(
            flattened_golds,
            flattened_preds,
            # labels=self.labels,
            output_dict=True,
            zero_division=0,
            labels=["0", "1"],
            target_names=["fair", "unfair"]
        )

    def evaluate(self,
                 samples: list[dict],
                 **kwargs) -> Union[float|dict]:
        """
        Predictions here should be the list of outputs made by the LLM (strings).
        sentences should be the matching sentences, the order should be the same of predictions. 
        """
        #self._check_pred_ref(predictions, sentences)
        self.samples = deepcopy(samples)
        sentences = [self.dataset.get_sentence_by_id(s["id_sentence"]) for s in self.samples]
        for s in self.samples:
            s["processed_output"] = self._preprocess_output(s["output"])
        predictions = [s["processed_output"] for s in self.samples]
        #print(f"predictions[:5] before preprocessing: {predictions[:5]}")
        predictions_processed = self._preprocess_predictions(predictions)
        #print(f"predictions[:5] after preprocessing: {predictions_processed[:5]}")
        #reference_tokens_dict = self.dataset.get_tokens_with_label(type_change="is_in_span", get_bio=True)
        gold_flattened = []
        pred_flattened = []
        for (prediction, sentence, sample) in zip(predictions_processed, sentences, self.samples):
            gold_tokens_with_label = sentence.get_tokens_with_label(type_change=self.type_change)
            pred_tokens_with_label = self._map_pred_to_tokens(prediction, sentence)
            if len(gold_tokens_with_label) != len(pred_tokens_with_label):
                raise Exception(f"found different lengths of tokens for sentence: {sentence.id}, gold: {len(gold_tokens_with_label)}, pred: {len(pred_tokens_with_label)} ")
            
            gold_flattened.extend(gold_tokens_with_label)
            pred_flattened.extend(pred_tokens_with_label)
            sample["pred_flattened"] = pred_tokens_with_label
            sample["gold_flattened"] = gold_tokens_with_label

        class_report = self._compute_classification_report(gold_flattened, pred_flattened)
            
        
        self.report["metrics"] = {}
        self.report["metrics"]["class_report"] = class_report
        self._dump_report()
        return class_report
    
class EvaluatorGeneration(Evaluator):

    def __init__(self,
                 **kwargs) -> None:
        rew_type = kwargs.pop("rew_type", "")
        super().__init__(**kwargs)
        self.rew_type = rew_type
        self.count_diff_tokens = rew_type == "io"

    

    def _get_data_csv(self, data_jsonl: list[dict]) -> Union[list[dict], None]:
        data_csv = []
        for sample in data_jsonl:
            to_copy = ["id_sentence", "sentence", "id_span", "span", "output"]
            new_dict = {f: sample[f] for f in to_copy}

            list_tokens = get_csv_tokens(sample["span_tok"])
            list_pred = split_list_for_csv(sample["prediction_tok"], "prediction_token")
            list_eval = split_list_for_csv(sample["evaluation"], "evaluation_token")

            new_dict.update(interleave_indexed_dicts(list_tokens, list_pred, list_eval))
            data_csv.append(new_dict)

        return data_csv
    
    @abc.abstractmethod
    def _process_tokens(self, tokenized_pred: list[str]) -> list[str]:
        pass

    def _preprocess_predictions(self,
                                predictions: list[str]):

        predictions_trim = [p.strip() for p in predictions]
        predictions_tokenized = []
        for pred in predictions_trim:
            tokenized_pred = nltk_tokenize(pred)
            tokenized_pred = self._process_tokens(tokenized_pred)
            predictions_tokenized.append(tokenized_pred)
        

        return predictions_tokenized

    def _convert_single_eval_to_0_1(self,
                                    eval: str
                            ) -> tuple[str, str]:
        """returns matching labels for an eval.
        It return first the pred, then the gold"""
        dict_eval = {
            "false_pos": ("1","0"),
            "false_neg": ("0","1"),
            "true_pos": ("1","1"),
            "true_neg": ("0","0")
        }

        eval_tuple = dict_eval.get(eval, None)
        if eval_tuple is None:
            raise ValueError(f"There is no value assigned for {eval} while evaluating")
        return eval_tuple
    
    def _convert_evals_to_0_1(self,
                              
                        ) -> tuple[list[str], list[str]]:
        """Given the samples after evaluation conducted on each token, 
        we convert it to two lists of classes that can go directly into classification_report.
        It returns first the prediction and then the gold."""

        y_pred = []
        y_gold = []

        for s in self.samples:
            token_evals = s["evaluation"]
            if isinstance(token_evals, list):
                #this is the normal case
                for eval in token_evals:
                    
                        pred, gold = self._convert_single_eval_to_0_1(eval)
                        y_pred.append(pred)
                        y_gold.append(gold)
            else: 
                #this is the instance of "diff_tokens" which is a str
                pass
        return y_pred, y_gold
                
    def evaluate(self,
                 samples: list[dict],
                 **kwargs) -> Union[float|dict]:
        
        self.samples = deepcopy(samples)
        sentence, spans, spans_tokens, predictions = self._get_sentences_sp_tokens_pred(self.samples)
        predictions_tokenized = self._preprocess_predictions(predictions)
        # print(predictions_tokenized[:5])
        # print([[s_t.token for s_t in spans_t] for spans_t in spans_tokens[:5]])

        counts = {
            "true_pos": 0,
            "true_neg": 0,
            "false_pos": 0,
            "false_neg": 0,
            "count_diff_tokens": 0
        }

        counts_type_errors = defaultdict(int)
        for (prediction_tok, span_tok, sample, span) in zip(predictions_tokenized, spans_tokens, self.samples, spans):
            sample["prediction_tok"] = prediction_tok
            sample["span_tok"] = span_tok
            #if self.count_diff_tokens and (len(prediction_tok) != len(span_tok)):
            if (len(prediction_tok) != len(span_tok)):
                tokenized_span = nltk_tokenize(span.span)
                if len(tokenized_span) != len(span_tok):
                    counts["count_diff_tokens"] += 1
                    sample["evaluation"] = "diff_tokens"
                    continue
                    #print(f"different length of tokenized span and span tokens for span: {span.id}, tokenized_span: {tokenized_span}, span_tok: {[t.token for t in span_tok]}")
                else:
                    prediction_tok= align_tok(prediction_tok, tokenized_span)

                # counts["count_diff_tokens"] += 1
                # sample["evaluation"] = "diff_tokens"
                
            sample["evaluation"] = []
            for (token_pred, token_gold) in zip(prediction_tok, span_tok):
                if self.use_dict_forms:
                    token_key = token_gold.token_lower
                    possible_forms = self.dataset.dict_token_forms.get(token_key, set())
                    possible_forms = possible_forms if len(possible_forms)>0 else None
                    (res, type_errors) = token_gold.is_valid_rewriting(token_pred, 
                                                        self.rew_type, 
                                                        possible_forms=possible_forms, 
                                                        return_type_error=True)
                else:
                    (res, type_errors) = token_gold.is_valid_rewriting(token_pred, 
                                                        self.rew_type,
                                                        return_type_error=True)
                counts[res] +=1
                for type_error in type_errors:
                    counts_type_errors[type_error] += 1
                sample["evaluation"].append(res)

        y_pred, y_gold = self._convert_evals_to_0_1()
        class_report = classification_report(y_gold, y_pred, labels=["0", "1"], output_dict=True)
        #TODO: standardizzare metrics per tutti evaluator
        self.report["metrics"] = {}
        self.report["metrics"]["counts"] = counts
        self.report["metrics"]["counts_type_errors"] = counts_type_errors
        self.report["metrics"]["class_report"] = class_report
        self._dump_report()
        return counts


class EvaluatorGenerationCV(EvaluatorGeneration):

    def __init__(self,
                 **kwargs) -> None:
        super().__init__(rew_type="cv",
                         **kwargs)

    
    
    def _process_tokens(self, tokenized_pred: list[str]) -> list[str]:
        return merge_slash_tokens(tokenized_pred)
    
    

class EvaluatorGenerationIO(EvaluatorGeneration):

    def __init__(self,
                 neomorpheme: Optional[str] = "ə",
                 **kwargs) -> None:
        super().__init__(rew_type="io",
                         **kwargs)
        self.neomorpheme = neomorpheme

    def _process_tokens(self, tokenized_pred: list[str]) -> list[str]:
        return [token.replace(self.neomorpheme, "<NEOM>") for token in tokenized_pred]
    
    



class EvaluatorGenerationIV(EvaluatorGeneration):

    def __init__(self,
                 neomorpheme: Optional[str] = "ə",
                 **kwargs) -> None:
        super().__init__(rew_type="iv",
                         **kwargs)
        self.neomorpheme = neomorpheme

    def _process_tokens(self, tokenized_pred: list[str]) -> list[str]:
        tokens_merged = merge_slash_tokens(tokenized_pred)
        tokens_merged_neom = [t.replace(self.neomorpheme, "<neom>") for t in tokens_merged]
        return tokens_merged_neom
    
class EvaluatorGenerationCO(Evaluator):

    #TODO: VEDERE SE PATH è CORRETTO
    PATH_CLASSIFIER = "./finetuning_gente/out/obscurative/checkpoint_4/"
    #PATH_CLASSIFIER = "FBK-MT/GeNTE-evaluator"
    PATH_TOKENIZER = "Musixmatch/umberto-commoncrawl-cased-v1"

    def __init__(self,
                 device: str = "cuda",
                 path_classifier: Optional[str] = None,
                 path_tokenizer: Optional[str] = None,
                 batch_size: int = 16,
                 path_llm_judge: Path = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.device = device
        self.path_classifier = path_classifier if path_classifier is not None else self.PATH_CLASSIFIER
        self.path_tokenizer = path_tokenizer if path_tokenizer is not None else self.PATH_TOKENIZER
        self.batch_size = batch_size
        self.path_llm_judge = path_llm_judge
        self.judgements = self._load_llm_judge()
        self.load_classifier()

    def _get_data_csv(self, data_jsonl: list[dict]) -> Union[list[dict], None]:
        return data_jsonl

    def _load_llm_judge(self) -> list[dict]:
        judgements = []
        if self.path_llm_judge is not None:
            path_llm_judge_model = self.path_llm_judge / f"{self.model_id}_judged.jsonl"
            if path_llm_judge_model.is_file():
                judgements = read_jsonl(path_llm_judge_model)
                print(f"Found judgement file with ({len(judgements)})")
                for j in judgements:
                    j["best_candidate_label"] = j["best_candidate"]["candidate"]
            else:
                print(f"Did not find the judged file at path: {path_llm_judge_model}")
        else:
            print(f"Path_llm_judge not valid ({self.path_llm_judge}), will not load it")
        return judgements

    def _add_judgements(self) -> None:

        found = 0
        for s in self.samples:
            matching_judgement = next((j for j in self.judgements if (
                (j["id_sentence"] == s["id_sentence"]) and (j["id_span"] == s["id_span"]))), None)
            if matching_judgement is not None:
                found+=1
            value_judgement = matching_judgement["best_candidate_label"] if matching_judgement is not None else None
            s["llm_judgement"] = value_judgement
        print(f"Out of {len(self.samples)}, we found {found} judgements")

    def _calculate_combined(self, sample: dict) -> int:
        pred_cls = sample["prediction"]
        llm_judge = sample["llm_judgement"]
        if llm_judge is not None:
            return 0 if (pred_cls==0 and llm_judge=="sì") else 1
        else:
            return pred_cls

    def load_classifier(self) -> None:
        print(f"Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.path_tokenizer, trust_remote_code=True)
        print(f"Loading classifier...")
        self.model = AutoModelForSequenceClassification.from_pretrained(self.path_classifier, trust_remote_code=True).to(self.device)
        self.model.eval()  # disable dropout etc. during inference
        print(f"Classifier succesfully loaded")

    def tokenize(self, texts: list) -> dict:
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.device)

    def classify(self, texts: list) -> list:
        """Classify texts in batches of self.batch_size, preserving input order."""
        predictions = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            inputs = self.tokenize(batch_texts)
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits
            batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
            predictions.extend(batch_predictions)
        return predictions

    def evaluate(self,
                 samples: list[dict],
                 **kwargs) -> Union[float, dict]:  #TODO: forse anche altro oltre a float

        self.samples = deepcopy(samples)
        for s in self.samples:
            s["processed_output"] = self._preprocess_output(s["output"])
        self._add_judgements()
        
        rewritten_spans = [sample["processed_output"] for sample in self.samples]
        id_spans = [sample["id_span"] for sample in self.samples]
        sentences = [self.dataset.get_sentence_by_id(sample["id_sentence"]) for sample in samples]
        sentences_with_rewritten_spans = [
            sentence.create_sentences_preselect(num_new=1,
                                                 type_alternative="CO",
                                                 return_type="sentence",
                                                 fixed_spans={id_span: rew_span})[0]
            for (rew_span, sentence, id_span) in zip(rewritten_spans, sentences, id_spans)
        ]
        # print(f"sentences_with_rewritten_spans: {sentences_with_rewritten_spans}")
        # here references is actually not important because we actually want all the predictions to be classified as 0
        prediction_labels = self.classify(sentences_with_rewritten_spans)
        for (s, pred_label) in zip(self.samples, prediction_labels):
            s["prediction"] = pred_label
            s["combined_prediction"] = self._calculate_combined(s)
            
        reference_labels = [0 for _ in prediction_labels]
        pred_combined = [s["combined_prediction"] for s in self.samples]
        #class_report = classification_report(reference_labels, prediction_labels, output_dict = True)
        class_report = classification_report(reference_labels, pred_combined, output_dict = True)

        self.report["metrics"] = {}
        self.report["metrics"]["class_report"] = class_report
        self._dump_report()
        return class_report