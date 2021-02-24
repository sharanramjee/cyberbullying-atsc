import re
import csv
import sklearn
import numpy as np
import aspect_based_sentiment_analysis as absa


def get_targets(tweet):
    handle_pattern = r'@(\w+)(?=[\s|:])'
    handles = re.findall(handle_pattern, tweet)
    return handles


def load_csv(filename):
    with open(filename, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        _ = next(csv_reader)   # CSV header
        tweets_targets = list()
        labels = list()
        for row in csv_reader:
            tweets_targets.append([row[0], get_targets(row[0])])
            labels.append(row[3])
        return tweets_targets, labels


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


def main():
    csv_path = 'data/target_test_tweets.csv'
    tweets_targets, labels = load_csv(csv_path)
    print('--- LOADED CSV ---')
    model = load_bert()
    print('--- LOADED MODEL ---')
    preds = list()
    count = 0
    for tweet, targets in tweets_targets:
        scores = score_sentiment(model, tweet, targets)
        preds.append(classify_sentiment(scores))
        count += 1
        if count % 1000:
            print(count, 'examples done')
    acc = sklearn.metrics.accuracy_score(labels, preds)
    print('Accuracy:', acc)


def demo():
    absa_model = load_bert()
    absa_text = 'The food was great, but the service was terrible.'
    absa_aspects = ['food']
    absa_scores = score_sentiment(absa_model, absa_text, absa_aspects)
    absa_labels = classify_sentiment(absa_scores)
    print(absa_labels)


if __name__ == '__main__':
    main()
