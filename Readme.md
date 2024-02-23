# Bert Sentiment Classification Model

We train a multi-lingual BERT and add a single-layer decoder that maps encoded latent to number of classes.

## Preprocessing

Given an input text, we convert it into lower classes and strip extra spaces. As the tokenizer can not recognize emojis, we demojize the sentence convert them to english sentences. This creates a problem I have not handled yet, given the time restriction as single sentence can contain multiple languages now.

## Training the Model

We use **google-bert/bert-base-multilingual-cased** as the encoder network and tokenizer. The tokenizer is trained over 104 languages and as the data is multilingual this provides us with a good starting point with a pretrained network. On top of the BERT encoder, we use a single layer LM Head as decoder network. Pretty simple but this provides us with a good enough starting point and this is the same as used by **TFBertForSequenceClassification**.

In the training script, we have a parameter if we want to train the encoder as well. It is False by default as we have a noisy dataset which is very small and can cause catastropichal forgetting or overfitting. So we freeze BERT and only train the LM Head.

Also, as the dataset is skewed with 80% positive examples, we use weighted over sampler to balance negative examples in the batch. Without this, the model tends to predict all positives which does acheive 80 percent accuracy but learns nothing.

I also provide additional functionality to include weighted loss for separate classes, but this tends to make model biased towards negative examples when combined with oversampler. So we omit it from our training script but it is left there in the dataset if need be in the future.

The models and logs are saved in the SAVE*DIR which is \_runs/models* by default.

## Inference

We can test the model on the using an inference script after the model is saved from the training script. I also provide a pretrained model available at: https://drive.google.com/file/d/1UJJf75vIvkdK9OOb5v0LdpKQNJMiMoqL/view?usp=sharing

## Setup

Clone this repository available on GitHub

```bash
git clone https://github.com/MuhammadArham-43/BertSentimentClassification.git
cd BertSentimentClassification
```

Setup a python environment and install required dependencies

```bash
python3 -m venv venv
source venv/bin/activate
```

Run training script

```python
python main.py --train_csv_path=<str> --num-classes=<int>
```

Inference

```python
python infer.py --text=<str> --trained-model-path=<str> --num-classes=2
```

## Sample Confusion Matrix

We get around 70% accuracy with the following final confusion matrix.
![Confusion Matrix](https://drive.google.com/file/d/1jVbclW1IkJhRSQm-yRskjnqaVZA5UDHU/view?usp=sharing)
