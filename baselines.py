import numpy as np
from transformers import pipeline
from utils import load_csv, save_npy, print_metrics


def load_bert_base():
    name = 'bert_base'
    model = pipeline('sentiment-analysis',
                     model='textattack/bert-base-uncased-SST-2')
    return model, name


def load_distilbert():
    name = 'distilbert'
    model = pipeline('sentiment-analysis')   # DistilBERT is the default
    return model, name


def load_twitter_roberta_base():
    name = 'twitter_roberta_base'
    model = pipeline('sentiment-analysis',
                     model='cardiffnlp/twitter-roberta-base-sentiment')
    return model, name


def load_bertweet():
    name = 'bertweet'
    model = pipeline('sentiment-analysis',
                     model='cardiffnlp/bertweet-base-sentiment')
    return model, name


def load_minilm():
    name = 'minilm'
    model = pipeline('sentiment-analysis',
                     model='microsoft/Multilingual-MiniLM-L12-H384')
    return model, name


def classify_sentiment(model, tweets, neg_label, name):
    n_chunks = 10
    chunk_size = len(tweets) // n_chunks
    chunks = [tweets[x:x+chunk_size] for x in range(0, len(tweets), chunk_size)]
    preds = list()
    for chunk_idx, chunk in enumerate(chunks):
        chunk_out = model(chunk)
        preds += [0 if out['label'] == neg_label else 1 for out in chunk_out]
        print('Chunk', chunk_idx, 'processed')
    save_npy(preds, name, 'preds/')
    return preds


def main():
    csv_path = 'data/target_test_tweets.csv'
    tweets, _, labels = load_csv(csv_path)

    bert_base, name = load_bert_base()
    print(name, 'loaded')
    preds = classify_sentiment(bert_base, tweets, 'LABEL_0', name)
    print_metrics(preds, labels, name)

    distilbert, name = load_distilbert()
    print(name, 'loaded')
    preds = classify_sentiment(distilbert, tweets, 'NEGATIVE', name)
    print_metrics(preds, labels, name)

    twitter_roberta_base, name = load_twitter_roberta_base()
    print(name, 'loaded')
    preds = classify_sentiment(twitter_roberta_base, tweets, 'LABEL_0', name)
    print_metrics(preds, labels, name)

    bertweet, name = load_bertweet()
    print(name, 'loaded')
    preds = classify_sentiment(bertweet, tweets, 'LABEL_0', name)
    print_metrics(preds, labels, name)
    
    # minilm = load_minilm()
    # print('MiniLM loaded')
    # preds = classify_sentiment(minilm, tweets, 'LABEL_0')
    # acc = compute_accuracy(preds, labels)
    # print('MiniLM Accuracy:', acc)


if __name__ == '__main__':
    main()
