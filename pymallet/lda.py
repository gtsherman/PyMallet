import argparse
import os
import random
import re
from collections import Counter
from io import StringIO

from pymallet import defaults, topicmodel
import numpy as np


token_regexes = {
    'en': r'(#|@)?(?!(\W)\2+)([a-zA-Z\_\-\'0-9\(-\@]{2,})',
    'zh': r'([\p{L}\p{M}]+|[\p{IsHan}]+)'
}


def _unified_interface(some_input, *args):
    if isinstance(some_input, StringIO):
        return some_input
    else:
        return open(some_input, *args)


def estimate_topics(docs_file, num_topics=defaults.NUM_TOPICS, alpha=defaults.ALPHA, beta=defaults.BETA,
                    iterations=defaults.NUM_ITERATIONS, stoplist=None, output_state=defaults.OUTPUT_STATE_FILE,
                    output_topic_keys=defaults.OUTPUT_TOPIC_KEYS_FILE, language=defaults.LANGUAGE):
    num_topics = num_topics
    doc_smoothing = alpha / num_topics
    word_smoothing = beta

    if stoplist is None:
        stoplist = set()

    word_counts = Counter()

    documents = []
    topic_totals = np.zeros(num_topics, dtype=int)

    print('Loading documents')
    docs_input = _unified_interface(docs_file)
    for line in docs_input:
        line = line.lower()

        doc_id, lang, line = line.split(' ', 2)
        tokens = [token.group(0) for token in re.finditer(token_regexes[language], line)]

        # remove stopwords
        tokens = [w for w in tokens if w not in stoplist]
        word_counts.update(tokens)

        doc_topic_counts = np.zeros(num_topics, dtype=int)

        documents.append(
            {"doc_id": doc_id, "original": line, "token_strings": tokens, "topic_counts": doc_topic_counts})

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
    model.sample(iterations)

    print('Printing state file')
    output_state_interface = _unified_interface(output_state, 'w')
    with output_state_interface as state_out:
        print('#doc source pos typeindex type topic', file=state_out)
        print('#alpha : {}'.format(' '.join([str(doc_smoothing) for _ in range(num_topics)])), file=state_out)
        print('#beta : {}'.format(str(word_smoothing)), file=state_out)

        for doc_i, document in enumerate(model.documents):
            for token_j in range(len(document.doc_tokens)):
                print(' '.join([str(doc_i), str(document.doc_id), str(token_j), str(document.doc_tokens[token_j]),
                                str(vocabulary[document.doc_tokens[token_j]]), str(document.doc_topics[token_j])]),
                      file=state_out)

    print('Printing keys file')
    output_keys_interface = _unified_interface(output_topic_keys, 'w')
    with output_keys_interface as key_out:
        model.print_all_topics(out=key_out)


def main():
    mallet_stoplists = ['de', 'en', 'fi', 'fr', 'jp', 'nl']

    options = argparse.ArgumentParser(
        description='Run latent Dirichlet allocation using collapsed Gibbs sampling.')
    options.add_argument('docs_file')
    options.add_argument('num_topics', nargs='?', type=int, default=defaults.NUM_TOPICS)
    options.add_argument('--output-state', type=str, default=defaults.OUTPUT_STATE_FILE)
    options.add_argument('--output-topic-keys', type=str, default=defaults.OUTPUT_TOPIC_KEYS_FILE)
    options.add_argument('--stoplist', help='Loads a stoplist file (a list of stopwords). Can also specify a Mallet '
                         'stoplist for the following languages: {}'.format(str(mallet_stoplists)))
    options.add_argument('--alpha', type=float, default=defaults.ALPHA)
    options.add_argument('--beta', type=float, default=defaults.BETA)
    options.add_argument('--iterations', type=int, default=defaults.NUM_ITERATIONS)
    options.add_argument('--language', type=str, choices=['en', 'zh'], default=defaults.LANGUAGE)
    args = options.parse_args()

    stoplist = set()
    if args.stoplist:
        if args.stoplist in mallet_stoplists:
            if args.stoplist != args.language:
                print('Warning: Stoplist language ({stoplang}) does not match overall language ({overlang})'.format(
                    args.stoplist, args.language))
            args.stoplist = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'stoplists',
                                         '{}.txt'.format(args.stoplist))
        try:
            with open(args.stoplist, encoding="utf-8") as stop_reader:
                for line in stop_reader:
                    line = line.rstrip()
                    stoplist.add(line)
            print('Loaded stoplist: {}'.format(args.stoplist))
        except FileNotFoundError:
            print('Unable to open stoplist file: {}'.format(args.stoplist))
    else:
        print('Warning: No stoplist selected. This may lead to suboptimal results.')

    estimate_topics(args.docs_file, args.num_topics, args.alpha, args.beta, args.iterations, stoplist,
                    args.output_state, args.output_topic_keys, language=args.language)


if __name__ == '__main__':
    main()
