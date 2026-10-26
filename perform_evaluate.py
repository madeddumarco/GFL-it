import argparse
from pathlib import Path
from utils.files_io import read_json, read_jsonl
from evaluation.evaluator import Evaluator, EvaluatorDetection, EvaluatorGenerationCO, EvaluatorGenerationCV, EvaluatorGenerationIO, EvaluatorGenerationIV


TASKS = ["detection", "rewriting_free", "rewriting_context"]
PREAMBLE_IDS = ["p1"]
REWRITING_TYPES = ["io", "cv", "co", "iv", "all"]

FIXED_KWARGS = {
    "type_change": "changed",
    "bio": False
}
def read_all_results(results_dir: Path) -> dict:
    """Function that reads the contents of the directory following the description of its 'help'.
       It returns a multi-level dictionary where the keys of the first level indicate the task.
       The second-level keys indicate the rewriting_type, the third level indicates the preamble_id, the fouth level keys indicate the model id.
       The values of the last is a dictionary composed by two keys 'samples' and 'recap' which respectily contain the content of the two files.
    """

    dict_results = {}
    for task_dir in results_dir.iterdir():
        if task_dir.is_dir():
            task_name = task_dir.name
            dict_results[task_name] = {}
            for rewriting_type_dir in task_dir.iterdir():
                if rewriting_type_dir.is_dir():
                    rewriting_type = rewriting_type_dir.name
                    dict_results[task_name][rewriting_type] = {}
                    for preamble_id_dir in rewriting_type_dir.iterdir():
                        if preamble_id_dir.is_dir():
                            preamble_id = preamble_id_dir.name
                            dict_results[task_name][rewriting_type][preamble_id] = {}
                            fewshot_dir = preamble_id_dir / "fewshot"
                            for model_dir in fewshot_dir.iterdir():
                                if model_dir.is_dir():
                                    model_id = model_dir.name
                                    dict_results[task_name][rewriting_type][preamble_id][model_id] = {}
                                    recap_file_path = None
                                    samples_file_path = None
                                    for file_path in model_dir.iterdir():
                                        if file_path.is_file() and file_path.suffix == ".json":
                                            recap_file_path = file_path
                                        elif file_path.is_file() and file_path.suffix == ".jsonl":
                                            samples_file_path = file_path
                                    if recap_file_path is not None:
                                        dict_results[task_name][rewriting_type][preamble_id][model_id]["recap"] = read_json(recap_file_path)
                                    if samples_file_path is not None:
                                        dict_results[task_name][rewriting_type][preamble_id][model_id]["samples"] = read_jsonl(samples_file_path)
    return dict_results



def print_results_lengths(all_results: dict,
                          tasks: list,
                          rewriting_types: list,
                          preambles: list) -> None:
    for task in tasks:
        for rewriting_type in rewriting_types:
            for preamble_id in preambles:
                if task in all_results and rewriting_type in all_results[task] and preamble_id in all_results[task][rewriting_type]:
                    for model_id, model_results in all_results[task][rewriting_type][preamble_id].items():
                        num_samples = len(model_results.get("samples", []))
                        print(f"Task: {task}, Rewriting Type: {rewriting_type}, Preamble ID: {preamble_id}, Model ID: {model_id}, Number of Samples: {num_samples}")
                else:
                    print(f"No results found for Task: {task}, Rewriting Type: {rewriting_type}, Preamble ID: {preamble_id}")


def expand_args(args_evaluator: dict,
                task: str,
                rewriting_type: str,
                args: argparse.Namespace):
    
    if task == "detection":
        args_evaluator.update({
            "type_change": FIXED_KWARGS["type_change"],
            "bio": FIXED_KWARGS["bio"]
        })
    elif task == "rewriting_context" and rewriting_type == "co":
        args_evaluator.update({
            "path_llm_judge": args.path_llm_judge_co
        })

    else:
        print(f"No expand args for this combo: {task}, {rewriting_type}")

def get_evaluator(**kwargs) -> Evaluator:

    task_name = kwargs.get("task_name")
    subtask_name = kwargs.get("subtask_name")

    if task_name == "detection":
        return EvaluatorDetection
    elif task_name == "rewriting_context" and subtask_name == "io":
        return EvaluatorGenerationIO
    elif task_name == "rewriting_context" and subtask_name == "iv":
        return EvaluatorGenerationIV
    elif task_name == "rewriting_context" and subtask_name == "cv":
        return EvaluatorGenerationCV
    elif task_name == "rewriting_context" and subtask_name == "co":
        return EvaluatorGenerationCO
    else:
        raise ValueError(f"no evaluator for {task_name}, {subtask_name}")
    
    

def evaluate_all_tasks(all_results: dict,
                        tasks: list,
                        rewriting_types: list,
                        preambles: list,
                        out_dir: Path,
                        use_dict_forms: bool,
                        args: argparse.Namespace) -> None:
    

    for task in tasks:
        for rewriting_type in rewriting_types:
            for preamble_id in preambles:
                if task in all_results and rewriting_type in all_results[task] and preamble_id in all_results[task][rewriting_type]:
                    for model_id, model_results in all_results[task][rewriting_type][preamble_id].items():
                        samples = model_results["samples"]
                        args_evaluator = {
                            "task_name": task,
                            "subtask_name": rewriting_type,
                            "model_id": model_id,
                            "out_dir": out_dir,
                            "save_samples": True,
                            "use_dict_forms": use_dict_forms
                        }
                        expand_args(args_evaluator, task, rewriting_type, args)
                        evaluator_class = get_evaluator(**args_evaluator)
                        print(f"Evaluator for {task}, {rewriting_type}, {model_id} has been created with class: {evaluator_class}")
                        evaluator : Evaluator = evaluator_class(**args_evaluator)
                        print(f"Evaluator instantied")
                        output_eval = evaluator.evaluate(samples)
                        print(f"got output: {output_eval}")
                else:
                    print(f"No results found for Task: {task}, Rewriting Type: {rewriting_type}, Preamble ID: {preamble_id}")


def main(args: argparse.Namespace) -> None:

    results_dir = args.results_dir
    out_dir = args.out_dir
    tasks = args.tasks
    preambles = args.preambles
    rewriting_types = args.rewriting_types
    use_dict_forms = args.use_dict_forms

    all_results = read_all_results(results_dir)
    

    #let's print all the lenghts of the results
    print_results_lengths(all_results, tasks, rewriting_types, preambles)
    
    evaluate_all_tasks(all_results=all_results,
                       tasks=tasks,
                       rewriting_types=rewriting_types,
                       preambles=preambles,
                       out_dir=out_dir,
                       use_dict_forms=use_dict_forms,
                       args=args)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", 
                        type=Path, 
                        required=True, 
                        help="""Directory that contains the outputs of the generative models to evaluate.
The directory is structured as follow: results_dir has children directories named with names of tasks from tasks. 
The directories of tasks have subdirectories named with rewriting types from rewriting_types. 
The rewriting_types directories named after the preamble used. 
Then there is a subdirectory named fewshot. 
Inside fewshot there is a directory for each model. 
This directory contains two files: one for recap and one for samples. 
The first is named recap_<timestamp>.json. The second is named recap_<timestamp>.json where <timestamp> is in format YYYY-MM-DD HH:mm:SS.
""")
    parser.add_argument("--out_dir", 
                        type=str, 
                        required=True, 
                        help="Directory that will be used to save the outputs of the evaluation.")
    parser.add_argument("--tasks", 
                        nargs="+",
                        choices=TASKS,
                        default=TASKS)
    parser.add_argument("--preambles", 
                        nargs="+",
                        choices=PREAMBLE_IDS,
                        default=PREAMBLE_IDS)
    parser.add_argument("--rewriting_types", 
                        nargs="*",
                        choices=REWRITING_TYPES,
                        default=REWRITING_TYPES)
    parser.add_argument("--use_dict_forms",
                        action="store_true")
    parser.add_argument("--path_llm_judge_co",
                        default=None,
                        required=False,
                        type=Path,
                        help="Path to the directory containing the llm as a judge for the task rewriting_context and rewriting_type co.")
    args = parser.parse_args()
    main(args)