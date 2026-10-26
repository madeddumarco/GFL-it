import abc
from utils.files_io import dump_jsonl, get_str_date, join_paths, create_dir
import torch
from transformers import AutoTokenizer
from typing import Union, Optional

class ModelBase(abc.ABC):
    def __init__(self, model_id: str, out_dir: str, tokenizer_id: str| None = None, gpu_num: str | None = None, do_create_dir: Optional[bool] = True
        ) -> None:

        self.model_id = model_id
        self.tokenizer_id = tokenizer_id or model_id
        self.timestamp = get_str_date()
        self.out_dir = out_dir
        self.model_out_dir = f"{self.out_dir}/{self.model_id.replace('/', ' ')}/{self.timestamp}/"
        self.device = self.select_device(gpu_num=gpu_num)
        if do_create_dir:
            create_dir(self.model_out_dir)
    
    def load_tokenizer(self, tokenizer_id: str) -> AutoTokenizer:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        return self.tokenizer

    def select_device(self, gpu_num: Union[str, None]) -> torch.device:
        if torch.cuda.is_available():
            if gpu_num is not None:
                device = torch.device(f"cuda:{gpu_num}")
                print(f"Using CUDA GPU {gpu_num}")
            else:
                device = torch.device("cuda")
                print("Using default CUDA GPU")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")  # GPU Apple (M1/M2/M3)
            print("Using Apple MPS GPU")
        else:
            if torch.version.hip:
                if  gpu_num is not None:
                    device = torch.device(f"cuda:{gpu_num}")
                    print(f"Using AMD ROCm GPU {gpu_num}")
                else:
                    device = torch.device("cuda")
                    print("Using default AMD ROCm GPU")
            else:
                device = torch.device("cpu")  # CPU fallback
                print("Using CPU")
        return device
    