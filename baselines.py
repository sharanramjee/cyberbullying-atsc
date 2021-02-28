from transformers import pipeline
from utils import load_csv, compute_accuracy


def load_bert_base():
    model = pipeline('sentiment-analysis',
                     model='nlptown/bert-base-multilingual-uncased-sentiment')
    return model


def load_distilbert():
    model = pipeline('sentiment-analysis')   # DistilBERT is the default
    return model


def load_roberta_base():
    model = pipeline('sentiment-analysis',
                     model='cardiffnlp/twitter-roberta-base-sentiment')
    return model


def load_bertweet():
    model = pipeline('sentiment-analysis',
                     model='cardiffnlp/bertweet-base-sentiment')
    return model


def load_t5_base():
    model = pipeline(
        'sentiment-analysis',
        model='mrm8488/t5-base-finetuned-span-sentiment-extraction')
    return model


def classify_sentiment(model, tweets):
    n_chunks = 10
    chunk_size = len(tweets) // n_chunks
    chunks = [tweets[x:x+chunk_size] for x in range(0, len(tweets), chunk_size)]
    preds = list()
    for chunk_idx, chunk in enumerate(chunks):
        chunk_out = model(chunk)
        chunk_preds = [1 if out['score'] >= 0.5 else 0 for out in chunk_out]
        preds += chunk_preds
        print('Chunk', chunk_idx, 'processed')
    return preds


def main():
    csv_path = 'data/target_test_tweets.csv'
    tweets, _, labels = load_csv(csv_path)

    bert_base = load_bert_base()
    print('BERT BASE loaded.')
    preds = classify_sentiment(bert_base, tweets)
    print('Inference complete.')
    acc = compute_accuracy(preds, labels)
    print('BERT BASE Accuracy:', acc)

    distilbert = load_distilbert()
    preds = classify_sentiment(distilbert, tweets)
    acc = compute_accuracy(preds, labels)
    print('DistilBERT Accuracy:', acc)

    roberta_base = load_roberta_base()
    preds = classify_sentiment(roberta_base, tweets)
    acc = compute_accuracy(preds, labels)
    print('RoBERTa BASE Accuracy:', acc)

    bertweet = load_bertweet()
    preds = classify_sentiment(bertweet, tweets)
    acc = compute_accuracy(preds, labels)
    print('BERTweet Accuracy:', acc)

    t5_base = load_t5_base()
    preds = classify_sentiment(t5_base, tweets)
    acc = compute_accuracy(preds, labels)
    print('T5 BASE Accuracy:', acc)


if __name__ == '__main__':
    main()
