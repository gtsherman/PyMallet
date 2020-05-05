import argparse
import random
import re
from collections import Counter

import numpy as np
import pyximport

pyximport.install()
import topicmodel


def load_stopwords(filename):
    stoplist = set()
    try:
        with open(filename, encoding="utf-8") as stop_reader:
            for line in stop_reader:
                line = line.rstrip()
                stoplist.add(line)
    except FileNotFoundError:
        print('Unable to open stoplist file: {}'.format(filename))
    return stoplist


if __name__ == '__main__':
    options = argparse.ArgumentParser(description='Run latent Dirichlet allocation using collapsed Gibbs sampling.')
    options.add_argument('docs_file')
    options.add_argument('num_topics', nargs='?', type=int, default=100)
    options.add_argument('--output-state', type=str, default='lda_model.state')
    options.add_argument('--output-topic-keys', type=str, default='lda_model.keys')
    options.add_argument('--stoplist', choices=['en'], default='en')
    options.add_argument('--extra-stopwords')
    options.add_argument('--no-stopping', action='store_true')
    options.add_argument('--alpha', type=float, default=0.05)
    options.add_argument('--beta', type=float, default=0.01)
    options.add_argument('--iterations', type=int, default=1000)
    args = options.parse_args()

    num_topics = args.num_topics
    doc_smoothing = args.alpha
    word_smoothing = args.beta

    if not args.no_stopping:
        print('Loading stoplist')
        stoplist = load_stopwords(args.stoplist)
        if args.extra_stopwords:
            stoplist = stoplist.union(load_stopwords(args.extra_stopwords))

    word_counts = Counter()

    documents = []
    word_topics = {}
    topic_totals = np.zeros(num_topics, dtype=int)

    print('Loading documents')
    for line in open(args.docs_file, encoding="utf-8"):
        # line = line.lower()

        doc_id, lang, line = line.split(' ', 2)
        tokens = [token.group(0) for token in re.finditer(r'(#|@)?(?!(\W)\2+)([a-zA-Z\_\-\'0-9\(-\@]{2,})', line)]

        # remove stopwords, short words, and upper-cased words
        tokens = [w for w in tokens if w not in stoplist]
        word_counts.update(tokens)

        doc_topic_counts = np.zeros(num_topics, dtype=int)

        documents.append(
            {"doc_id": doc_id, "original": line, "token_strings": tokens, "topic_counts": doc_topic_counts})

    # Now that we're done reading from disk, we can count the total number of words.

    vocabulary = list(word_counts.keys())
    vocabulary_size = len(vocabulary)
    word_ids = {w: i for (i, w) in enumerate(vocabulary)}
    smoothing_times_vocab_size = word_smoothing * vocabulary_size

    word_topics = np.zeros((len(vocabulary), num_topics), dtype=int)

    print('Initializing')
    for document in documents:
        tokens = document["token_strings"]
        doc_topic_counts = document["topic_counts"]

        doc_tokens = np.ndarray(len(tokens), dtype=int)
        doc_topics = np.ndarray(len(tokens), dtype=int)
        topic_changes = np.zeros(len(tokens), dtype=int)

        for i, w in enumerate(tokens):
            word_id = word_ids[w]
            topic = random.randrange(num_topics)

            doc_tokens[i] = word_id
            doc_topics[i] = topic

            # Update counts:
            word_topics[word_id][topic] += 1
            topic_totals[topic] += 1
            doc_topic_counts[topic] += 1

        document["doc_tokens"] = doc_tokens
        document["doc_topics"] = doc_topics
        document["topic_changes"] = topic_changes

    sampling_dist = np.zeros(num_topics, dtype=float)
    topic_normalizers = np.zeros(num_topics, dtype=float)
    for topic in range(num_topics):
        topic_normalizers[topic] = 1.0 / (topic_totals[topic] + smoothing_times_vocab_size)

    model = topicmodel.TopicModel(num_topics, vocabulary, doc_smoothing, word_smoothing)

    print('Adding documents to model')
    for document in documents:
        c_doc = topicmodel.Document(document["doc_id"], document["doc_tokens"], document["doc_topics"],
                                    document["topic_changes"], document["topic_counts"])
        model.add_document(c_doc)

    print('Estimating topics')
    model.sample(args.iterations)

    print('Printing state file')
    with open(args.output_state, 'w') as state_out:
        print('#doc source pos typeindex type topic', file=state_out)
        print('#alpha : {}'.format(' '.join([str(doc_smoothing) for _ in range(num_topics)])), file=state_out)
        print('#beta : {}'.format(str(word_smoothing)), file=state_out)

        for doc_i, document in enumerate(model.documents):
            for token_j in range(len(document.doc_tokens)):
                print(' '.join([str(doc_i), str(document.doc_id), str(token_j), str(document.doc_tokens[token_j]),
                                str(vocabulary[document.doc_tokens[token_j]]), str(document.doc_topics[token_j])]),
                      file=state_out)

    print('Printing keys file')
    with open(args.output_topic_keys, 'w') as key_out:
        model.print_all_topics(out=key_out)
