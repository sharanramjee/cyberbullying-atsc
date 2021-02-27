from transformers import pipeline
from utils import load_csv, compute_accuracy


def load_bert_base():
    model = pipeline('sentiment-analysis',
                     model='nlptown/bert-base-multilingual-uncased-sentiment')
    return model


def load_roberta_base():
    model = pipeline('sentiment-analysis',
                     model='cardiffnlp/twitter-roberta-base-sentiment')
    return model


def classify_sentiment(model, tweets):
    outputs = model(tweets)
    preds = [1 if output['label'] == 'POSITIVE' else 0 for output in outputs]
    return preds


def main():
    csv_path = 'data/target_test_tweets.csv'
    tweets, _, labels = load_csv(csv_path)

    bert_base = load_bert_base()
    preds = classify_sentiment(bert_base, tweets)
    acc = compute_accuracy(preds, labels)
    print('BERT BASE Accuracy:', acc)


if __name__ == '__main__':
    main()
