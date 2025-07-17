# VLM Preference Finetuning with ORPO

This repository demonstrates a proof-of-concept for finetuning a Vision Language Model (VLM) on a preference dataset using the ORPO (Monolithic Preference Optimization without Reference Model) method. Specifically, it finetunes the LLaVA-1.6 7B model on the RLAIF-V-Dataset.

The full training logs and metrics can be accessed via [Weights & Biases (wandb)](https://wandb.ai/cowboy_bebop/llava-qlora-orpo).

## Key Notes

* **Training Environment:**

  * Initial attempts to run this project entirely within a Google Colab notebook proved infeasible with the free tier due to computational limits.
  * Default training parameters require approximately 24GB of VRAM and complete training in roughly 2 hours on an RTX4070 GPU.
  * Only 10% of the original dataset was used due to computational constraints (server rental).

* **Implementation Highlights:**

  * This method computes attention layers once for the prompt and image, then reuses these computed weights for both "chosen" and "rejected" responses. This approach significantly accelerates training and evaluation.
  * LoRA was preferred over QLoRA due to feature limitations in QLoRA, which prevented the efficient reuse of precomputed weights. Consequently, LoRA proved faster and easier for this application.

## Repository Structure

* `code/config.py`: Configuration parameters for training and evaluation.
* `code/dataloader_helper.py`: Utility functions for dataset loading and preprocessing.
* `code/orpo_helper.py`: Functions for efficient forward passes and ORPO-specific loss calculations.
* `code/ORPO_LLAVA.ipynb`: Main notebook to load data, initialize models, and train adapters.
* `code/validation_hosting.ipynb`: Notebook for loading trained adapters, performing final evaluations, quantazing and saving the best model.

## Results

| Name        | ORPO Loss (λ = 10) | Loss SFT  | Loss OR  |
|-------------|--------------------|-----------|----------| 
| base_model  | 14.3907            | 7.3019    | 0.7089   |
| last        | 13.1701            | 7.2270    | 0.5943   |
| best_exp_1  | 13.1854            | 7.0621    | 0.6123   |
| best_exp_2  | 13.5973            | 7.4392    | 0.6158   |

- base_model  
  The base model before finetuning.
- last  
  The latest checkpoint of second experiment.
- best_exp_1  
  The best model according to validation of the 1 experiment.
- best_exp_2  
  The best model according to validation of the 2 experiment.

As you can see, all the finetuned models have lower ORPO loss.  
The main difference between experiment 1 and 2 is that 2 uses larger λ value for training. 
Which you can see in the loss (expriment 1 has lowest SFT score since OR loss played less role). 
Also we can see that the validation wasn't representative enough since the last model actually outperforms the best model in experiment 2.  
Overall we can conclude that even short finetuning improved the results of the model. 

I am in the middle of running quantization via vLLM's llm compressor and hosting the model with vLLM and guardio UI.

## Conclusions

The finetuned adapters exhibit lower ORPO loss, particularly a reduced OR loss, indicating a higher probability of generating "chosen" tokens compared to "rejected" tokens. While improvements from initial finetuning were modest, further training with additional data or prolonged training duration is expected to yield greater performance gains.
