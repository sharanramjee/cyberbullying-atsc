from transformers import pipeline
from utils import load_csv, compute_accuracy


def load_bert_base():
    model = pipeline('sentiment-analysis',
                     model='textattack/bert-base-uncased-SST-2')
    return model


def load_distilbert():
    model = pipeline('sentiment-analysis')   # DistilBERT is the default
    return model


def load_twitter_roberta_base():
    model = pipeline('sentiment-analysis',
                     model='cardiffnlp/twitter-roberta-base-sentiment')
    return model


def load_bertweet():
    model = pipeline('sentiment-analysis',
                     model='cardiffnlp/bertweet-base-sentiment')
    return model


def load_minilm():
    model = pipeline('sentiment-analysis',
                     model='microsoft/Multilingual-MiniLM-L12-H384')
    return model


def classify_sentiment(model, tweets, neg_label):
    n_chunks = 10
    chunk_size = len(tweets) // n_chunks
    chunks = [tweets[x:x+chunk_size] for x in range(0, len(tweets), chunk_size)]
    preds = list()
    for chunk_idx, chunk in enumerate(chunks):
        chunk_out = model(chunk)
        preds += [0 if out['label'] == neg_label else 1 for out in chunk_out]
        print('Chunk', chunk_idx, 'processed')
    return preds


def main():
    csv_path = 'data/target_test_tweets.csv'
    tweets, _, labels = load_csv(csv_path)

    bert_base = load_bert_base()
    print('BERT BASE loaded')
    preds = classify_sentiment(bert_base, tweets, 'LABEL_0')
    acc = compute_accuracy(preds, labels)
    print('BERT BASE Accuracy:', acc)

    distilbert = load_distilbert()
    print('DistilBERT loaded')
    preds = classify_sentiment(distilbert, tweets, 'NEGATIVE')
    acc = compute_accuracy(preds, labels)
    print('DistilBERT Accuracy:', acc)

    twitter_roberta_base = load_twitter_roberta_base()
    preds = classify_sentiment(twitter_roberta_base, tweets, 'LABEL_0')
    print('Twitter RoBERTa loaded')
    acc = compute_accuracy(preds, labels)
    print('Twitter RoBERTa BASE Accuracy:', acc)

    # bertweet = load_bertweet()
    # print('BERTweet loaded')
    # preds = classify_sentiment(bertweet, tweets, 'LABEL_0')
    # acc = compute_accuracy(preds, labels)
    # print('BERTweet Accuracy:', acc)
    #
    # minilm = load_minilm()
    # print('MiniLM loaded')
    # preds = classify_sentiment(minilm, tweets, 'LABEL_0')
    # acc = compute_accuracy(preds, labels)
    # print('MiniLM Accuracy:', acc)


if __name__ == '__main__':
    main()
