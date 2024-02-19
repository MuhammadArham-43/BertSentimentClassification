import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import ReviewDataset
from models import BertSentimentClassifier

from tqdm import tqdm


if __name__ == "__main__":
    TRAIN_DATA_CSV_PATH = "/home/evobits/arham/data/bertData/train_data.csv"
    MODEL_PATH = "google-bert/bert-base-multilingual-cased"
    CKPT_PATH = "/home/evobits/arham/trainBert/runs/models/best.pth"
    
    dataset = ReviewDataset(csv_path=TRAIN_DATA_CSV_PATH)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertSentimentClassifier(num_classes=2, model_path=MODEL_PATH, freeze_bert=True)    
    model.load_state_dict(torch.load(CKPT_PATH)["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    
    num_correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    for batch in tqdm(dataloader):
            texts, labels = batch
            encoded_texts = model.encode_texts(texts)
            
            labels = labels.to(DEVICE)
            encoded_texts = encoded_texts.to(DEVICE)
    
            predictions = model(encoded_texts)
            predicted_classes = torch.argmax(predictions, dim=1)
            all_predictions.extend(predicted_classes.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            num_correct += (predicted_classes == labels).sum().item()
            total += labels.shape[0]
            
    print("Accuracy: ", (num_correct / total) * 100)
    print("F1 Score: ", f1_score(all_labels, all_predictions))
    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    
    plt.savefig("runs/cmatrix.png")
