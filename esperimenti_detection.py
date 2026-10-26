import os
import logging
import wandb
import torch
from dotenv import load_dotenv
from huggingface_hub import login as hf_login
import argparse
from detection.ModelDetection import VanillaBERTDetectionModel
from detection.config import ModelConfig, GRID_SEARCH_PARAMS

load_dotenv()
HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

# ---- Training Mode Toggle ----
# Set to True for hyperparameter tuning with Optuna, False for regular training
# ENABLE_HYPERPARAMETER_TUNING = False
# ENABLE_LORA = True
# ---- Data Configuration ----
DATA_PATH = "Data/gfl-it.json"  # TODO: Set your dataset path
# TYPE_CHANGE = "changed"  # Options: "changed", "is_in_span"
#GET_BIO = True

# ---- Model Configuration ----
MODEL_CONFIG = ModelConfig(
    learning_rate=3.58995e-05,
    weight_decay=0.0004934,
    epochs=3, 
)

# ---- Model and Output Configuration ----
# MODEL_NAME = "osiria/deberta-base-italian"
# OUTPUT_DIR = "./detection_results"
# Automatically use CUDA if available, otherwise CPU
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Detection Task Configuration ----
LABELS_BIO = ["O", "B-UNFAIR", "I-UNFAIR"]
LABELS_BINARY = ["0", "1"]

# ---- Logging Configuration ----
#WANDB_LOGGING = True  # Set to False to disable wandb logging

# Configure root logger with proper handler
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # Force reconfiguration even if already configured
)
logger = logging.getLogger(__name__)

# Reduce verbosity of transformers library logging
transformers_logger = logging.getLogger('transformers')
transformers_logger.setLevel(logging.WARNING)

# ---- Authentication and Setup ----
def setup_external_services(wandb_logging: bool):
    """Initialize authentication for HuggingFace and wandb."""
    logger.info("Setting up external services...")
    
    # HuggingFace login
    if HF_TOKEN:
        try:
            hf_login(token=HF_TOKEN)
            logger.info("HuggingFace login successful")
        except Exception as e:
            logger.warning(f"HuggingFace login failed: {e}")
    else:
        logger.warning("HF_TOKEN not set in environment")
    
    # wandb login and initialization
    if wandb_logging and WANDB_API_KEY:
        try:
            wandb.login(key=WANDB_API_KEY)
            logger.info("wandb login successful")
        except Exception as e:
            logger.warning(f"wandb login failed: {e}")
    elif wandb_logging:
        logger.warning("WANDB_API_KEY not set, wandb logging disabled")


def main(args):


    model_id = args.model_id
    tokenizer_id = args.tokenizer_id
    out_dir = args.out_dir
    hyperparameter_tuning = args.hyperparameter_tuning
    enable_lora = args.enable_lora
    type_change = args.type_change
    bio = args.bio
    print(f"bio: {bio}")
    #gpu_num = args.gpu_num
    wandb_logging = args.wandb_logging
    num_folds = args.num_folds
    path_grid_search = args.path_grid_search

    # ---- Grid Search Configuration ----
    # Define the search space for grid search when enabled
    grid_search = GRID_SEARCH_PARAMS if hyperparameter_tuning else None

    """Main training pipeline"""
    
    logger.info("=" * 80)
    logger.info("Starting GFL Detection Training")
    logger.info("=" * 80)    
    logger.info(f"Device: automatically selected by model") #TODO: improve
    #TODO: fix this with gpu_num
    # if gpu_num == "cuda":
    #     logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    #     logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")    
    logger.info(f"Hyperparameter tuning: {hyperparameter_tuning}")
    if hyperparameter_tuning:
        logger.info(f"Grid search parameters: {grid_search}")
    if path_grid_search:
        logger.info(f"Loading best hyperparameters from: {path_grid_search}")
    logger.info(f"Using LoRa: {enable_lora}")
    logger.info(f"Model: {model_id}")
    logger.info(f"Learning rate: {MODEL_CONFIG.learning_rate}")
    logger.info(f"Epochs: {MODEL_CONFIG.epochs}")
    logger.info(f"Weight decay: {MODEL_CONFIG.weight_decay}")
    
    # Setup external services
    setup_external_services(wandb_logging)
    
    # Initialize wandb if enabled
    if wandb_logging and WANDB_API_KEY:
        wandb.init(
            project="gfl-detection",
            name="vanilla-bert-detection",  # TODO: Change to the name of the strategy
            config={
                "model": model_id,
                "learning_rate": MODEL_CONFIG.learning_rate,
                "epochs": MODEL_CONFIG.epochs,
                "weight_decay": MODEL_CONFIG.weight_decay,
                "hyperparameter_tuning": hyperparameter_tuning,
                "num_folds": num_folds,
                "label_type": type_change,
                "lora_enabled": enable_lora
            },
            tags=["vanilla-bert", "detection"], # TODO: Change to the name of the strategy
        )
    
    try:
        # Create model instance
        logger.info("Instantiating VanillaBERTDetectionModel...") # TODO: Change to the name of the model
        model = VanillaBERTDetectionModel(
            model_config=MODEL_CONFIG,
            grid_search=grid_search,
            path_grid_search=path_grid_search,
            bio=bio,
            labels=LABELS_BIO if bio else LABELS_BINARY,
            num_folds=num_folds,
            mode="train",
            hyperp_tuning=hyperparameter_tuning,
            lora=enable_lora,
            wandb_logging=wandb_logging,
            model_id=model_id,
            out_dir=out_dir,
            tokenizer_id=tokenizer_id, #TODO: this could be changed to a specific tokenizer_id in case they do not match
            #gpu_num=gpu_num,
        )
        logger.info("Model instantiated successfully")
        
        # Train model
        logger.info("Starting training...")
        model.train_model(
            path=DATA_PATH,
            type_change=type_change,
            get_bio=bio,
        )
        
        logger.info("=" * 80)
        logger.info("Training completed successfully!")
        logger.info(f"Results saved to: {model.model_out_dir}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise
    
    finally:
        # Finish wandb run if enabled
        if wandb_logging and WANDB_API_KEY:
            wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id",
                        type=str,
                        required=True,
                        help="Id of the hf model to finetune")
    
    parser.add_argument("--tokenizer_id",
                        type=str,
                        help="Id of the hf tokenizer to use")
    
    parser.add_argument("--out_dir",
                        type=str,
                        default="./detection_results",
                        help="Id of the hf model to finetune")
    
    parser.add_argument("--hyperparameter_tuning",
                        action="store_true",
                        help="Do hyperparameter tuning or not")
    
    parser.add_argument("--enable_lora",
                        action="store_true",
                        help="Use lora adapters or full parameters")
    
    parser.add_argument("--type_change",
                        type=str,
                        choices=["changed", "is_in_span"],
                        required=True,
                        help="Get tokens based on either them being present in a span ('is_in_span') or if they actually change when rewritten ('changed')")
    
    parser.add_argument("--bio",
                        action="store_true",
                        help="Wheter or not to perform BIO")
    
    #TODO: remove this, accelerate cannot work with this
    # parser.add_argument("--gpu_num",
    #                     type=str,
    #                     help="In case the machine has multiple gpus, specify the one to use")
    
    parser.add_argument("--num_folds",
                        type=int,
                        required=False,
                        default=5,
                        help="Number of folds for cross validation")
    
    parser.add_argument("--wandb_logging",
                        action="store_true",
                        help="Log results into wandb, if enabled it loads the wandb key from env")
    
    parser.add_argument("--path_grid_search",
                        type=str,
                        default=None,
                        help="Path to previous grid search results to load best hyperparameters from")
    
    args = parser.parse_args()
    if args.tokenizer_id is None:
        args.tokenizer_id = args.model_id
    main(args)