import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, RMSprop
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from datasets import ReviewDataset
from models import BertSentimentClassifier

from tqdm import tqdm
import os
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument(
        "--train-csv-path",
        help="path to training data csv file",
        type=str,
        required=True
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        required=True
    )
    parser.add_argument(
        "--batch-size",
        default=32,
        type=int
    )
    parser.add_argument(
        "--num-epochs",
        default=100,
        type=int
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cpu"
    )
    parser.add_argument(
        "--save-dir",
        default="runs/models"
    )
    parser.add_argument(
        "--train-backbone",
        type=bool,
        default=False,
        help="Whether to train the BERT model or only LM Head."
    )
    
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_arguments()
    
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs
    NUM_CLASSES = args.num_classes
    TRAIN_DATA_CSV_PATH = args.train_csv_path
    MODEL_PATH = "google-bert/bert-base-multilingual-cased"
    DEVICE = args.device
    
    SAVE_DIR = args.save_dir
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    
    dataset = ReviewDataset(csv_path=TRAIN_DATA_CSV_PATH)
    
    class_oversampler = dataset.get_weighted_sampler()
    train_dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=class_oversampler if class_oversampler else None
    )
    
    class_weights = torch.tensor([1,1.2], dtype=torch.float32).to(DEVICE)
    criterion = CrossEntropyLoss(weight=class_weights)
    
    model = BertSentimentClassifier(num_classes=NUM_CLASSES, model_path=MODEL_PATH, freeze_bert=not args.train_backbone)    
    model.to(DEVICE)
    
    optimizer_params = [
        {'params': model.bert.parameters(), 'lr': 5e-4},
        {'params': model.decoder.parameters(), 'lr': 1e-2}
    ]
    optimizer = RMSprop(optimizer_params)
    
    writer = SummaryWriter("runs/logs")
    iterations = 0
    for epoch in range(NUM_EPOCHS):
        
        running_loss = 0
        for batch in tqdm(iter(train_dataloader)):
            iterations += 1
            texts, labels = batch
            encoded_texts = model.encode_texts(texts)
            
            labels = labels.to(DEVICE)
            encoded_texts = encoded_texts.to(DEVICE)
            
            predictions = model(encoded_texts)
            
            loss = criterion(predictions, labels)
            running_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            writer.add_scalar("Loss / Iteration", loss.item(), iterations)
            
        avg_epoch_loss = running_loss / len(train_dataloader)
        writer.add_scalar("Loss / Epoch", avg_epoch_loss, epoch)        
        print(f"Epoch {epoch+1} / {NUM_EPOCHS} || Loss: {avg_epoch_loss}")
        
        if (epoch + 1) % 10 == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epochs": epoch
            }, os.path.join(SAVE_DIR, f"model_epoch{epoch+1}.pth"))    