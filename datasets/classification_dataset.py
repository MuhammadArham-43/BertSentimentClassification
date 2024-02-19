from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
from emoji import demojize
import numpy as np
import torch

LABEL_MAP = {
    "negative": 0,
    "positive": 1,
}


class ReviewDataset(Dataset):
    def __init__(self, csv_path: str):
        super().__init__()
        self.data = pd.read_csv(csv_path)
        self.data = self.preprocess_dataset(self.data)
        self.data.reset_index(drop=True, inplace=True)
        self.texts = self.data["comment_description"]
        self.labels = self.data["sentiment"]
        assert len(self.texts) == len(self.labels), "Number of comments should equal the number of labels"
    
    def __len__(self):
        return len(self.texts)
    
    def preprocess_dataset(self, df: pd.DataFrame):
        df['sentiment'] = df['sentiment'].str.lower().str.strip()
        target_labels_lower_stripped = [label.lower().strip() for label in LABEL_MAP.keys()]
        filtered_df = df[df['sentiment'].isin(target_labels_lower_stripped)]
        return filtered_df
    
    
    def preprocess_text(self, text):
        cleaned_text = demojize(text)
        return cleaned_text
    
    def get_weighted_sampler(self, oversampling_ratio: float = None):
        class_counts = np.bincount(torch.tensor(self.labels == "negative"))
        num_positive_samples = class_counts[0].item()
        num_negative_samples = class_counts[1].item()
        oversampling_ratio = oversampling_ratio if oversampling_ratio else num_positive_samples // num_negative_samples
        weights = [1] * len(self)  # Initialize all weights to 1.0
        for idx, target in enumerate(self.labels):
            if target == 'negative':
                weights[idx] = oversampling_ratio
                
        return WeightedRandomSampler(weights, len(self), replacement=True)
        
    def get_class_weights(self) -> torch.tensor:
        weights = np.array([0] * len(LABEL_MAP.keys()))
        for key in LABEL_MAP.keys():
            count = (self.labels == key).sum()
            weights[LABEL_MAP[key]] = count
            
        weights = weights / np.sum(weights)
        class_weights = 1 / weights
        return torch.tensor(class_weights, dtype=torch.float32)
    
    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index].strip().lower()
        text = self.preprocess_text(text)
        label = LABEL_MAP[label]
        return text, label



if __name__ == "__main__":
    data = ReviewDataset("/home/evobits/arham/data/bertData/train_data.csv")
    weights = data.get_class_weights()
    print(weights)
    data.get_weighted_sampler()
    
    class_oversampler = data.get_weighted_sampler()
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        data,
        batch_size=128,
        sampler=class_oversampler if class_oversampler else None
    )
    
    for batch in enumerate(train_dataloader):
        texts, labels = batch
        print(texts)
        print("\n\n*****\n\n")
        print(labels)
        neg_samples = (labels == 0).sum()
        pos_samples = (labels == 1).sum()
        print(neg_samples, pos_samples)
    