import re
import csv
import numpy as np
from sklearn.metrics import precision_score, recall_score
from ner import load_ner_model, get_ner_targets


def get_username_targets(tweet):
    handle_pattern = r'@(\w+)(?=[\s|:])'
    handles = re.findall(handle_pattern, tweet)
    return handles


def create_target_csv(in_csv, out_csv):
    ner_model = load_ner_model()
    with open(out_csv, 'w', newline='', encoding='utf-8') as out_csv_file:
        csv_writer = csv.writer(out_csv_file)
        with open(in_csv, encoding='utf-8') as in_csv_file:
            csv_reader = csv.reader(in_csv_file, delimiter=',')
            header = next(csv_reader)  # CSV header
            csv_writer.writerow(header)
            ex_idx = 0
            for row in csv_reader:
                ner_targets = get_ner_targets(ner_model, row[0])
                if len(ner_targets) == 0:
                    csv_writer.writerow(row)
                else:
                    tweet = row[0]
                    for target in ner_targets:
                        try:
                            targ_idx = tweet.index(target)
                            tweet = tweet[:targ_idx] + '@' + tweet[targ_idx:]
                        except:
                            continue
                    csv_writer.writerow([tweet] + row[1:])
                ex_idx += 1
                print('Example', ex_idx, 'processed')


def load_csv(filename):
    with open(filename, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        _ = next(csv_reader)   # CSV header
        tweets = list()
        targets = list()
        labels = list()
        for row in csv_reader:
            tweets.append(row[0])
            tweet_targets = get_username_targets(row[0])
            if len(tweet_targets) == 0:
                targets.append([row[0]])
            else:
                targets.append(tweet_targets)
            labels.append(int(row[3]))
        return tweets, targets, labels


def combine_csv(csv1, csv2, out_csv):
    with open(out_csv, 'w', newline='', encoding='utf-8') as out_csv_file:
        csv_writer = csv.writer(out_csv_file)
        with open(csv1, encoding='utf-8') as csv1_file:
            csv1_reader = csv.reader(csv1_file, delimiter=',')
            header = next(csv1_reader)  # CSV header
            csv_writer.writerow(header)
            for row in csv1_reader:
                csv_writer.writerow(row)
        with open(csv2, encoding='utf-8') as csv2_file:
            csv2_reader = csv.reader(csv2_file, delimiter=',')
            _ = next(csv2_reader)  # CSV header
            for row in csv2_reader:
                csv_writer.writerow(row)


def save_npy(arr, name, dir_name='preds/'):
    np.save(dir_name + name + '.npy', arr)


def load_npy(filename):
    arr = np.load(filename)
    return arr


def compute_accuracy(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    correct = np.sum(preds == labels)
    total = len(labels)
    acc = correct / total
    return acc


def compute_precision(preds, labels):
    # preds = np.array(preds)
    # labels = np.array(labels)
    # tp = np.sum((preds == 1) * (labels == 1))
    # fp = np.sum((preds == 1) * (labels == 0))
    # return tp / (tp + fp)
    return precision_score(labels, preds)


def compute_recall(preds, labels):
    # preds = np.array(preds)
    # labels = np.array(labels)
    # tp = np.sum((preds == 1) * (labels == 1))
    # fn = np.sum((preds == 0) * (labels == 1))
    # return tp / (tp + fn)
    return recall_score(labels, preds)


def compute_f1(preds, labels):
    precision = compute_precision(preds, labels)
    recall = compute_recall(preds, labels)
    return 2 * precision * recall / (precision + recall)


def print_metrics(preds, labels, name):
    acc = compute_accuracy(preds, labels)
    prec = compute_precision(preds, labels)
    rec = compute_recall(preds, labels)
    f1 = compute_f1(preds, labels)
    print(name, 'accuracy:', acc)
    print(name, 'precision:', prec)
    print(name, 'recall:', rec)
    print(name, 'F1:', f1)


if __name__ == '__main__':
    csv_one = 'data/no_target_test_clean.csv'
    csv_two = 'data/ner_no_target_test_clean.csv'
    combined_csv = 'data/all_test_clean.csv'
    combine_csv(csv_one, csv_two, combined_csv)
