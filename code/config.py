MODEL_ID          = "llava-hf/llava-v1.6-mistral-7b-hf"
DATASET_NAME      = "openbmb/RLAIF-V-Dataset"
WANDB_PROJECT     = "llava-qlora-orpo"
OUTPUT_DIR        = "../logs/"

# USE_QLORA         = False 
# QLORA had a lot of issues with the Mistral model
# SO I dropped it for now

TRAIN_BATCH_SIZE  = 1 
VAL_BATCH_SIZE    = 6
TEST_BATCH_SIZE   = 6
GRAD_ACC_STEPS    = 4          # effective batch = TRAIN_BATCH_SIZE × GRAD_ACC_STEPS
EPOCHS            = 1
LEARNING_RATE     = 2e-4
WARMUP_RATIO      = 0.03

VAL_RATIO         = 0.05      
TEST_RATIO        = 0.20  

LORA_R            = 8  
LORA_ALPHA        = 16 

LORA_DROPOUT      = 0.05
ORPO_LAMBDA       = 10

LOG_EVERY_STEPS   = 4
VAL_EVERY_STEPS   = 200

DEVICE = "cuda"

# Maximum number of tokens for the answer
MAX_ANSWER_TOKENS = 128 