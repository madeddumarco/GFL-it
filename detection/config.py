from dataclasses import dataclass
from itertools import product

@dataclass
class ModelConfig():
    learning_rate: float = 3.58995e-05
    epochs: int = 5 
    weight_decay: float = 0.0004934 


def _get_grid_search(learning_rate_params: list[float],
                     weight_decay_params: list[float]) -> list[dict]:
    
    grid_search = []
    for lr, wd in product(learning_rate_params, weight_decay_params):
        grid_search.append({
            "learning_rate": lr,
            "weight_decay": wd
        })
    return grid_search

LEARNING_RATE_PARAMS = [1e-5, 4e-5, 5e-5]
WEIGHT_DECAY_PARAMS = [0.01, 0.001, 0.0001]
GRID_SEARCH_PARAMS = _get_grid_search(LEARNING_RATE_PARAMS, WEIGHT_DECAY_PARAMS)