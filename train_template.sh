#!/bin/bash
#SBATH --job-name=twitter_sentiment_analysis
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<your nus email>@comp.nus.edu.sg
#SBATCH --gpus=h100-47:1
#SBATCH --partition=gpu-long

source 'your_path_to_your_virtualenv'/bin/activate
# Name and notes optional
export WANDB_API_KEY='your_wandb_api_key'
export WANDB_NAME="Twitter sentiment analysis"
export WANDB_NOTES="Init run."

which python
python --version
nvidia-smi

# python train_bert_tweet.py \
#     --epochs 4 \
#     --model_name "vinai/bertweet-base"

python train.py \
    --epochs 10 \
    --data_path "./archive/train_with_features.csv" \
    --test_data_path "./archive/test_with_features.csv" \
    --gpt_feature \
    --model_name "google-bert/bert-large-cased" \
    --max_length 512