from typing import List, Optional
from utils.files_io import read_txt, join_paths

class PromptMaker:

    DIR_DATA_PROMPT = "./Data/generation/prompt/"

    #da capire se tokenization fatta fuori o qua dentro

    def _read_preamble(self, preamble: str, task: str, rewriting_type: str) -> str:
        path_preamble = join_paths(self.DIR_DATA_PROMPT, "preamble", task, rewriting_type, f"{preamble}.txt")
        text_preamble = read_txt(path_preamble, return_list=False, strip=True) 
        return text_preamble 
    
    def _read_instructions(self, task: str) -> str:
        path_instructions = join_paths(self.DIR_DATA_PROMPT, "instructions", f"{task}.txt")
        text_instructions = read_txt(path_instructions, return_list=False, strip=True) 
        return text_instructions 
    
    def _read_fewshot_examples(self, fewshot: bool, task: str, rewriting_type: str) -> str:
        
        if fewshot:
            str_task = "da identificare" if task=="detection" else "da riscrivere"
            preamble_fewshot = f"Prima ti verrano mostrati degli esempi di come svolgere il tuo compito e poi ti verrà mostrato il testo {str_task}\n\n###Esempi\n\n"
            path_fewshot = join_paths(self.DIR_DATA_PROMPT, "fewshot", task, f"{rewriting_type}.txt")
            text_fewshot = read_txt(path_fewshot, return_list=False, strip=True) 
            return preamble_fewshot + text_fewshot 
        else:
            return ""

    def _string_input_span_texts(self, input_span_text: str | None) -> str:
        if input_span_text is not None:
            preamble_input_span_texts = "[Span]: " #TODO: scrivere il vero preamble
            input_span_texts_str = preamble_input_span_texts + input_span_text
            return input_span_texts_str 
        else:
            return ""
        
    def _read_last_line(self, task: str) -> str:
        path_last_line = join_paths(self.DIR_DATA_PROMPT, "last_line", f"{task}.txt")
        text_last_line = read_txt(path_last_line, return_list=False, strip=True) 
        return text_last_line 

    def _get_text_input(self, sentence: str) -> str:
        return f"[Testo]: {sentence}" 
    

    def get_prompt_text(self,
                   task: str,
                   rewriting_type: str,
                   sentence_text: str,
                   preamble_id: str,        
                   fewshot: bool = False,
                   input_span_text: Optional[str] = None, 
                   )-> str : 
        
        if task not in ["detection", "rewriting_free", "rewriting_context"]:
            raise ValueError(f"Task must be one between: 'detection', 'rewriting_free', 'rewriting_context'. Got {task} instead.")
        
        if task in ["rewriting_context"] and rewriting_type not in ["io", "cv", "co", "iv"]:
            raise ValueError(f"Rewriting type for {task} must be one between: 'io', 'cv', 'co', 'iv'. Got {rewriting_type} instead.")
        
        if task in ["detection", "rewriting_free"] and rewriting_type != "all":
            raise ValueError(f"Rewriting type for {task} must be 'all'. Got {rewriting_type} instead.")
        
        if preamble_id not in ["p1", "p2", "p3"]:  
            raise ValueError(f"Preamble must be one between: 'p1', 'p2', 'p3'. Got {preamble_id} instead.")
        
        if task not in ["rewriting_context", "rewriting_free"]:
            input_span_text = None

        text_preamble = self._read_preamble(preamble=preamble_id, 
                                            task=task,
                                            rewriting_type=rewriting_type) #this is basically the preamble of the prompt
        text_fewshot = self._read_fewshot_examples(rewriting_type=rewriting_type,
                                                   fewshot=fewshot,
                                                   task=task)

        text_input_spans = self._string_input_span_texts(input_span_text)
        text_last_line = self._read_last_line(task=task)
        text_input = self._get_text_input(sentence_text)
        text_instructions = self._read_instructions(task)
        pre_text_input = "###Testo da identificare:" if task == "detection" else "###Testo da riscrivere:"
        #prompt_text = text_preamble + text_fewshot + text_detected_spans + text_input + text_last_line #questo è da verificare
        #prompt_text = "\n\n".join([text_preamble, text_fewshot, text_detected_spans, text_input, text_last_line])
        #TODO: fix detected spans
        prompt_format = "{preamble}\n\n{text_instructions}\n\n{fewshot_examples}\n\n{pre_text_input}\n\n{text_input}\n{text_input_spans}\n{last_line}"
        
        prompt_text = prompt_format.format(preamble=text_preamble, 
                                           text_instructions=text_instructions,
                                           fewshot_examples=text_fewshot, 
                                           text_input_spans=text_input_spans, 
                                           pre_text_input=pre_text_input,
                                           text_input=text_input,
                                           last_line=text_last_line)
        
        return prompt_text

    __call__ = get_prompt_text