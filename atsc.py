import numpy as np
from utils import load_csv, save_npy, print_metrics
import aspect_based_sentiment_analysis as absa


def load_bert():
    model = absa.load()
    return model


def score_sentiment(model, text, targets):
    try:
        output = list(model(text, aspects=targets))
        scores = [out.scores for out in output]
    except:
        print('Skipped')
        scores = [[0, 0, 1]]
    return scores


def classify_sentiment(scores):
    positive_score = np.mean([score[2] for score in scores])
    negative_score = np.mean([score[1] for score in scores])
    pred_label = 1       # Not cyberbullying
    if negative_score >= positive_score:
        pred_label = 0   # Cyberbullying
    return pred_label


def predict(model, tweets, targets):
    count = 0
    preds = list()
    for tweet, target in zip(tweets, targets):
        scores = score_sentiment(model, tweet, target)
        preds.append(classify_sentiment(scores))
        count += 1
        print('Example', count, 'processed')
    return preds


def demo():
    absa_model = load_bert()
    absa_text = 'The food was great, but the service was terrible.'
    absa_aspects = ['food']
    absa_scores = score_sentiment(absa_model, absa_text, absa_aspects)
    absa_labels = classify_sentiment(absa_scores)
    print(absa_labels)


def main():
    csv_path = 'data/target_test_tweets.csv'
    tweets, targets, labels = load_csv(csv_path)
    print('--- LOADED CSV ---')
    model = load_bert()
    print('--- LOADED MODEL ---')
    preds = predict(model, tweets, targets)
    save_npy(preds, 'ada_bert', 'preds/')
    print('--- SAVED PREDS ---')
    print_metrics(preds, labels, 'ada_bert')


if __name__ == '__main__':
    main()
