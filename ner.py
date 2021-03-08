from transformers import pipeline


def load_ner_model():
    model = pipeline('ner')
    return model


def get_ner_targets(model, tweet):
    entities = model(tweet)
    targets = list()
    for entity in entities:
        if entity['entity'] == 'I-PER':
            if entity['word'][:2] == '##':
                if len(targets) == 0:
                    targets.append(entity['word'][2:])
                else:
                    targets[-1] = targets[-1] + entity['word'][2:]
            else:
                targets.append(entity['word'])
    return targets


if __name__ == '__main__':
    ner_model = load_ner_model()
    text = 'German for Dummies (Unabridged) [Unabridged Nonfiction] - Edward Swick | #Languages |â€¦ https://t.co/Sz85iMzWN8'
    ner_targets = get_ner_targets(ner_model, text)
    print(ner_targets)
