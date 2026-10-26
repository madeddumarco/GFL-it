from typing import Union, Optional
from GFLDataLoader.GFLDataset import Sentence, Span
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
from .Prompt import PromptMaker
from utils.files_io import dump_jsonl, get_str_date, join_paths, create_dir, dump_json
from tqdm import tqdm
from pathlib import Path
from pathlib import Path
from utils.ModelBase import ModelBase

class ModelRewriting(ModelBase):

    def __init__(self, 
                 **kwargs
        ) -> None:

        super().__init__(do_create_dir=False,
                         **kwargs)
        # self.model_id = model_id
        # self.tokenizer_id = tokenizer_id or model_id
        # self.timestamp = get_str_date()
        # self.out_dir = out_dir
        # self.model_out_dir = f"{self.out_dir}/{self.model_id.replace('/',' ')}/{self.timestamp}/"
        # self.device = self.select_device(gpu_num=gpu_num)
        # create_dir(self.model_out_dir)

    def _get_dir_to_save(**kwargs) -> Path:
        pass

    def _get_recap(self, **kwargs) -> dict:
        return {
            "general_info": {
                "model_id": self.model_id,
                "tokenizer_id": self.tokenizer_id,
                "timestamp": self.timestamp,
                "out_dir": self.out_dir,
                **kwargs
            }
        }

    def _save_recap(self, out_dir: Union[str,Path], **kwargs) -> None:
        recap = self._get_recap(**kwargs)
        filename = join_paths(out_dir, f"recap_{self.timestamp}.json")
        dump_json(recap, filename)

    
    def _save_samples(self, 
                      samples: list[dict], 
                      rewritten: list[str],
                      dir_to_save: Union[str, Path]) -> None:
        if len(samples) != len(rewritten):
            raise ValueError("Length of samples and rewritten outputs must match.")
        
        data_to_save = []
        for sample, rewrite in zip(samples, rewritten):
            sentence = sample["sentence"]
            input_span = sample.get("input_span", None)
            text_span = getattr(input_span, "span", None)
            span_id = getattr(input_span, "id", -1)
            original_text = sentence.sentence if isinstance(sentence, Sentence) else sentence
            dict_sample = {
                "id_sentence": getattr(sentence, "id", -1),  # Use sentence ID if available, else -1
                "sentence": original_text,
                "output": rewrite,
                
            }
            if input_span is not None:
                dict_sample.update({"id_span": span_id,
                                    "span": text_span})
            data_to_save.append(dict_sample)
        
        filename = join_paths(dir_to_save, f"samples_{self.timestamp}.jsonl")
        dump_jsonl(data_to_save, filename)

    def rewrite_cv(self, sentences: list[Union[Sentence, str]], do_evaluate: bool = True) -> list[str]:
        pass

    def rewrite_io(self, sentences: list[Union[Sentence, str]], do_evaluate: bool = True) -> list[str]:
        pass

    def rewrite_co(self, sentences: list[Union[Sentence, str]], do_evaluate: bool = True) -> list[str]:
        pass


class ModelRewritingDecoder(ModelRewriting):
    prompt_maker = PromptMaker()

    def __init__(
        self,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.generation_config = generation_config or GenerationConfig(
            do_sample=False,
            temperature=0.0,
            max_new_tokens=256
        )
        self.model = self.load_model(self.model_id)
        self.tokenizer = self.load_tokenizer(self.tokenizer_id)
        

    def _get_dir_to_save(self, **kwargs) -> Path:
        out_dir = self.out_dir
        approach = "generative"
        task = kwargs.get("task", "unknown_task")
        rewriting_type = kwargs.get("rewriting_type", "unknown_rewriting_type")
        preamble_id = kwargs.get("preamble_id", "unknown_preamble")
        fewshot = "fewshot" if kwargs.get("fewshot", False) else "zeroshot"
        model_id = self.model_id.replace('/', ' ')
        dir_to_save = join_paths(out_dir, approach, task, rewriting_type, preamble_id, fewshot, model_id)
        create_dir(dir_to_save)
        return dir_to_save

    def _get_recap(self, **kwargs) -> dict:
        kwargs_for_recap = {
            "task": kwargs.pop("task", "unknown_task"),
            "rewriting_type": kwargs.pop("rewriting_type", "unknown_rewriting_type"),
            "preamble_id": kwargs.pop("preamble_id", "unknown_preamble"),
            "fewshot": kwargs.pop("fewshot", False),
            "generation_config": kwargs.pop("generation_config", self.generation_config.to_dict() if self.generation_config else None),
            "prompt_example": kwargs.pop("prompt_example", ""),
            "num_samples": kwargs.pop("num_samples", -1),
        }
        recap = super()._get_recap(**kwargs)
        recap["generative_info"] = {
            **kwargs_for_recap
        } 
        return recap
        
    # ------------------------------------------------------------------ #
    #  Model / tokenizer loading                                           #
    # ------------------------------------------------------------------ #

    
    
    def load_model(self, model_id: str) -> AutoModelForCausalLM:
        """Load a decoder-only model in inference mode."""
        if self.device.type == "cuda" and (self.device.index is None or self.device.index == "None"):
            self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            #token=self.token_hf
        ) 
        else: 
            self.model = AutoModelForCausalLM.from_pretrained(model_id, 
                                                              dtype="auto", 
                                                              #token=self.token_hf
                                                              ).to(self.device)
            self.model.eval()
        return self.model

    def load_tokenizer(self, tokenizer_id: str) -> AutoTokenizer:
        """Load a tokenizer, setting pad token to eos if absent and padding to left."""
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # required for decoder-only batch generation
        return self.tokenizer

    def cleanup(self):
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
    # ------------------------------------------------------------------ #
    #  Chat template interface                                             #
    # ------------------------------------------------------------------ #

    def build_chat_messages(
        self,
        user_content: str,
        system_prompt: Optional[str] = None,
    ) -> list[dict]:
        """
        Build a messages list in the standard chat format.

        Parameters
        ----------
        user_content:
            The user turn content.
        system_prompt:
            Optional system prompt. Falls back to ``self.system_prompt``.

        Returns
        -------
        list[dict]
            A list of role/content dicts ready for ``apply_chat_template``.
        """
        messages = []
        system = system_prompt if system_prompt else ""
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_content})
        return messages

    def apply_chat_template(
        self,
        messages: list[dict],
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Apply the tokenizer's chat template to a messages list.

        Parameters
        ----------
        messages:
            List of role/content dicts (e.g. from ``build_chat_messages``).
        add_generation_prompt:
            Whether to append the assistant turn opener so the model
            continues as the assistant.

        Returns
        -------
        str
            The rendered prompt string.
        """
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    # def format_prompts(
    #     self,
    #     sentences: list[str],
    #     prompt_args: dict,
    #     system_prompt: Optional[str] = None,
    #     detected_spans: Optional[list[list[str]]] = None,
    # ) -> list[str]:
    #     """
    #     Convert a list of sentences into chat-templated prompt strings.

    #     Parameters
    #     ----------
    #     sentences:
    #         Raw strings or ``Sentence`` objects to format as user turns.
    #     system_prompt:
    #         Optional system prompt override for this batch.

    #     Returns
    #     -------
    #     list[str]
    #         Rendered prompt strings, one per sentence.
    #     """
    #     prompts = []
    #     for (i, sentence) in enumerate(sentences):
    #         detected_spans_sentence = detected_spans[i] if detected_spans is not None else None
    #         user_content = self.prompt_maker(sentence=sentence, 
    #                                          detected_spans_sentence=detected_spans_sentence,
    #                                          **prompt_args)
    #         messages = self.build_chat_messages(user_content, system_prompt)
    #         prompts.append(self.apply_chat_template(messages))
    #     return prompts
    def format_prompts(
        self,
        samples: list[dict],
        prompt_args: dict,
        system_prompt: Optional[str] = None
    ) -> list[str]:
        """
        Convert a list of sentences into chat-templated prompt strings.

        Parameters
        ----------
        samples:
            Dicts containg sentence and sdetected_span (this last one is optional)
        system_prompt:
            Optional system prompt override for this batch.

        Returns
        -------
        list[str]
            Rendered prompt strings, one per sentence.
        """
        prompts = []
        for (i, sample) in enumerate(samples):
            span_obj = sample.get("input_span", None)
            span_text = getattr(span_obj, "span", None)
            sentence_text = sample["sentence"].sentence
            user_content = self.prompt_maker(sentence_text=sentence_text, 
                                             input_span_text=span_text,
                                             **prompt_args)
            messages = self.build_chat_messages(user_content, system_prompt)
            prompts.append(self.apply_chat_template(messages))
        return prompts

    # ------------------------------------------------------------------ #
    #  Core inference interface                                            #
    # ------------------------------------------------------------------ #

    def tokenize(
        self,
        texts: list[str],
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
    ) -> dict:
        """Tokenize a batch of strings and move tensors to the model device."""
        kwargs = dict(padding=padding, truncation=truncation, return_tensors="pt")
        if max_length is not None:
            kwargs["max_length"] = max_length
        encoded = self.tokenizer(texts, **kwargs)
        return {k: v.to(self.device) for k, v in encoded.items()}

    @torch.inference_mode()
    def generate(
        self,
        inputs: Union[list[str], dict],
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> list[str]:
        """
        Generate text for a batch of prompts.

        Parameters
        ----------
        inputs:
            Either a list of raw strings (tokenized internally) or a
            pre-built dict of tensors as returned by ``tokenize()``.
        generation_config:
            Override the instance-level config for this call only.
        **kwargs:
            Extra keyword arguments forwarded to ``model.generate()``.

        Returns
        -------
        list[str]
            Decoded output strings with special tokens stripped.
        """
        if isinstance(inputs, list):
            inputs = self.tokenize(inputs)

        

        config = generation_config if generation_config is not None else self.generation_config
        input_length = inputs["input_ids"].shape[1]

        # print(f"Generating with config: {config}")
        # print(f"Generation kwargs: {kwargs}")

        
        output_ids = self.model.generate(
            **inputs,
            generation_config=config,
            **kwargs,
        )
        new_token_ids = output_ids[:, input_length:]
        old_token_ids = output_ids[:, :input_length]
        return self.decode(new_token_ids)

    def decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        """Decode a batch of token-id tensors into strings."""
        return self.tokenizer.batch_decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )
    
    @staticmethod
    def get_samples(sentences: list[Sentence], 
                    detected_spans: list[list[Span]] | None = None) -> list[dict]:
        """Restituisce una lista di samples. Per i task hanno anche span come input (riscritture), vengono messi anche gli span.
Visto il for sugli span, vengono automaticamente filtrati i testi senza span."""
        samples = []
        if detected_spans is not None:
            assert(len(sentences) == len(detected_spans)), f"Len of sentences ({len(sentences)}) does not match len of detected spans: {len(detected_spans)}"
            for (sentence, detected_spans_sentence) in zip(sentences, detected_spans):
                for span in detected_spans_sentence:
                    samples.append({
                        "sentence": sentence,
                        "input_span": span
                    })
        else:
            samples = [{"sentence": s} for s in sentences]
            
        return samples
    
    def generate_from_sentences(
        self,
        sentences: list[Sentence],
        prompt_args: dict,
        batch_size: int = 8,
        system_prompt: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
        input_spans: Optional[list[list[Span]]] = None,
        **kwargs,
    ) -> list[str]:
        """
        Apply the chat template to sentences and generate in batches.

        Parameters
        ----------
        sentences:
            Raw sentences or ``Sentence`` objects forming the user turn.
        system_prompt:
            Optional system prompt override for this batch.
        generation_config:
            Override the instance-level config for this call only.
        batch_size:
            Number of sentences to process per forward pass.

        Returns
        -------
        list[str]
            Generated output strings, in the same order as input.
        """

        task=prompt_args.get("task")
        rewriting_type=prompt_args.get("rewriting_type")
        preamble_id=prompt_args.get("preamble_id")
        fewshot=prompt_args.get("fewshot")

        if (task in ["rewriting_context", "rewriting_free"]) and (input_spans is None):
            raise Exception(f"For task: {task}, you have to pass input_spans that is not None")

        #sentence_texts = [sentence.sentence if isinstance(sentence, Sentence) else sentence for sentence in sentences]
        samples = self.get_samples(sentences, input_spans)
        # prompts = self.format_prompts(sentences=sentence_texts, 
        #                               prompt_args=prompt_args, 
        #                               system_prompt=system_prompt,
        #                               detected_spans=detected_spans)

        prompts = self.format_prompts(samples=samples, 
                                      prompt_args=prompt_args, 
                                      system_prompt=system_prompt
                                     )
        
        #print(prompts[:5])
        
        rewritings = []
        time_start = get_str_date()
        for i in tqdm(range(0, len(prompts), batch_size),
                      desc="Generating from Sentences"):
            batch = prompts[i : i + batch_size]
            rewritings.extend(
                self.generate(batch, generation_config=generation_config, **kwargs)
            )
        time_end = get_str_date()

        dir_to_save = self._get_dir_to_save(task=task,
                                            rewriting_type=rewriting_type,
                                            preamble_id=preamble_id,
                                            fewshot=fewshot)
        self._save_samples(samples=samples,
                           rewritten=rewritings,
                           dir_to_save=dir_to_save) 
        
        self._save_recap(out_dir=dir_to_save, 
                         task=task,
                        rewriting_type=rewriting_type,
                        preamble_id=preamble_id,
                        fewshot=fewshot,
                        generation_config=generation_config.to_dict() if generation_config else self.generation_config.to_dict(),
                        prompt_example=prompts[0] if len(prompts) else "",
                        num_samples=len(samples),
                        time_start = time_start,
                        time_end=time_end
                        )
        
        return rewritings
    # ------------------------------------------------------------------ #
    #  Abstract stubs (implemented by subclasses)                         #
    # ------------------------------------------------------------------ #

    def rewrite_cv(self, sentences: list[Union[Sentence, str]], do_evaluate: bool = True) -> list[str]:
        raise NotImplementedError

    def rewrite_io(self, sentences: list[Union[Sentence, str]], do_evaluate: bool = True) -> list[str]:
        raise NotImplementedError

    def rewrite_co(self, sentences: list[Union[Sentence, str]], do_evaluate: bool = True) -> list[str]:
        raise NotImplementedError