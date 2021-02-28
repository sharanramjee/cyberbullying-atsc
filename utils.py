import re
import csv
import numpy as np


def get_targets(tweet):
    handle_pattern = r'@(\w+)(?=[\s|:])'
    handles = re.findall(handle_pattern, tweet)
    return handles


def load_csv(filename):
    with open(filename, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        _ = next(csv_reader)   # CSV header
        tweets = list()
        targets = list()
        labels = list()
        for row in csv_reader:
            tweets.append(row[0])
            targets.append(get_targets(row[0]))
            labels.append(int(row[3]))
        return tweets, targets, labels


def compute_accuracy(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    correct = np.sum(preds == labels)
    total = len(labels)
    acc = correct / total
    return acc

def compute_precision(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    tp = np.sum((preds == 1) * (labels == 1))
    fp = np.sum((preds == 1) * (labels == 0))
    return tp / (tp + fp)

def compute_recall(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    tp = np.sum((preds == 1) * (labels == 1))
    fn = np.sum((preds == 0) * (labels == 1))
    return tp / (tp + fn)

def compute_f1(preds, labels):
    precision = compute_precision(preds, labels)
    recall = compute_recall(preds, labels)
    return 2 * precision * recall / (precision + recall)
