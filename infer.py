import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import ReviewDataset
from models import BertSentimentClassifier

import os
from tqdm import tqdm
import argparse



def parse_arguments():
        parser = argparse.ArgumentParser()
        
        parser.add_argument(
                "--text",
                required=True,
                type=str,
        )
        parser.add_argument(
                "--trained-model-path",
                required=True,
                type=str
        )
        parser.add_argument(
                "--num-classes",
                required=True,
                type=int
        )
        parser.add_argument(
                "--device",
                default="cpu",
                type=str
        )

        args = parser.parse_args()
        return args

if __name__ == "__main__":
        args = parse_arguments()       

        text = args.text
        MODEL_PATH = "google-bert/bert-base-multilingual-cased"
        CKPT_PATH = args.trained_model_path
        DEVICE = args.device
        
        model = BertSentimentClassifier(num_classes=args.num_classes, model_path=MODEL_PATH, freeze_bert=True)    
        model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE)["model_state_dict"])
        model.to(DEVICE)
        model.eval()
        
        encoded_text = model.encode_texts([text]).to(DEVICE)
        with torch.no_grad():
                predictions = model(encoded_text)
        predicted_class = torch.argmax(predictions, dim=1).squeeze(0).item()
        print("postive" if predicted_class == 1 else "negative")