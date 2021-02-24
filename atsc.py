import aspect_based_sentiment_analysis as absa


def load_bert():
    model = absa.load()
    return model


def score_sentiment(model, text, aspect):
    score, _ = model(text, aspects=[aspect, '.'])
    return score


if __name__ == '__main__':
    absa_model = load_bert()
    absa_text = 'The food was great, but the service was terrible.'
    absa_aspect = 'service'
    absa_sentiment = score_sentiment(absa_model, absa_text, absa_aspect)
    print(absa_sentiment.sentiment)
