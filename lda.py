import argparse
import cProfile
import math
import pstats
import random
import re
import sys
from collections import Counter
from timeit import default_timer as timer

import topicmodel

import numpy as np
import pyximport

pyximport.install()

word_pattern = re.compile("\w[\w\-\']*\w|\w")

options = argparse.ArgumentParser(description='Run latent Dirichlet allocation using collapsed Gibbs sampling.')
options.add_argument('docs_file')
options.add_argument('num_topics', nargs='?', type=int, default=100)
options.add_argument('--output-state', type=str, default='lda_model.state')
options.add_argument('--output-topic-keys', type=str, default='lda_model.keys')
options.add_argument('--stoplist', choices=['en'], default='en')
options.add_argument('--alpha', type=float, default=0.5)
options.add_argument('--beta', type=float, default=0.01)
args = options.parse_args()

num_topics = args.num_topics
doc_smoothing = args.alpha
word_smoothing = args.beta

stoplist = set()
with open("stoplists/{}.txt".format(args.stoplist), encoding="utf-8") as stop_reader:
    for line in stop_reader:
        line = line.rstrip()
        stoplist.add(line)

word_counts = Counter()

documents = []
word_topics = {}
topic_totals = np.zeros(num_topics, dtype=int)

for line in open(args.docs_file, encoding="utf-8"):
    # line = line.lower()

    doc_id, lang, line = line.split(' ', 2)
    tokens = word_pattern.findall(line)

    # remove stopwords, short words, and upper-cased words
    tokens = [w for w in tokens if not w in stoplist and len(w) >= 3 and not w[0].isupper()]
    word_counts.update(tokens)

    doc_topic_counts = np.zeros(num_topics, dtype=int)

    documents.append({"doc_id": doc_id, "original": line, "token_strings": tokens, "topic_counts": doc_topic_counts})

# Now that we're done reading from disk, we can count the total number of words.

vocabulary = list(word_counts.keys())
vocabulary_size = len(vocabulary)
word_ids = {w: i for (i, w) in enumerate(vocabulary)}
smoothing_times_vocab_size = word_smoothing * vocabulary_size

word_topics = np.zeros((len(vocabulary), num_topics), dtype=int)

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


def profile():
    model = topicmodel.TopicModel(50, len(vocabulary), doc_smoothing, word_smoothing)
    document = documents[0]

    for document in documents:
        c_doc = topicmodel.Document(document["doc_tokens"], document["doc_topics"], document["topic_changes"],
                                    document["topic_counts"])
        model.add_document(c_doc)

    # model.sample(10)

    # cProfile.runctx("topicmodel.sample_doc(doc_tokens, doc_topics, topic_changes, doc_topic_counts, word_topics,
    # topic_totals, sampling_dist, topic_normalizers, doc_smoothing, word_smoothing, smoothing_times_vocab_size,
    # num_topics)", globals(), locals(), "topics.prof")
    cProfile.runctx("model.sample(10)", globals(), locals(), "topics.prof")

    stats = pstats.Stats("topics.prof")
    stats.strip_dirs().sort_stats("time").print_stats()


def sample(num_iterations):
    start = timer()

    for iteration in range(num_iterations):

        for document in documents:
            doc_topic_counts = document["topic_counts"]
            doc_tokens = document["doc_tokens"]
            doc_topics = document["doc_topics"]
            topic_changes = document["topic_changes"]

            # Pass the document to the fast C code
            topicmodel.sample_doc(doc_tokens, doc_topics, topic_changes, doc_topic_counts, word_topics, topic_totals,
                                  sampling_dist, topic_normalizers, doc_smoothing, word_smoothing,
                                  smoothing_times_vocab_size, num_topics)

        if iteration % 10 == 0:
            end = timer()
            print(end - start)
            start = timer()


def entropy(p):
    # make sure the vector is a valid probability distribution
    p = p / np.sum(p)

    result = 0.0
    for x in p:
        if x > 0.0:
            result += -x * math.log2(x)

    return result


def print_topic(topic):
    sorted_words = sorted(zip(word_topics[:, topic], vocabulary), reverse=True)

    for i in range(20):
        w = sorted_words[i]
        print("{}\t{}".format(w[0], w[1]))


def print_all_topics():
    for topic in range(num_topics):
        sorted_words = sorted(zip(word_topics[:, topic], vocabulary), reverse=True)
        print(" ".join([w for x, w in sorted_words[:20]]))


def write_state(writer):
    writer.write("Doc\tWordID\tWord\tTopic\tCounts\tChanges\n")

    for doc, document in enumerate(documents):
        doc_tokens = document["doc_tokens"]
        doc_topics = document["doc_topics"]
        topic_changes = document["topic_changes"]

        doc_length = len(doc_tokens)

        for i in range(doc_length):
            word_id = doc_tokens[i]
            word = vocabulary[word_id]
            topic = doc_topics[i]
            changes = topic_changes[i]

            writer.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(doc, word_id, word, topic, word_counts[word], changes))


# profile()

model = topicmodel.TopicModel(num_topics, vocabulary, doc_smoothing, word_smoothing)

for document in documents:
    c_doc = topicmodel.Document(document["doc_id"], document["doc_tokens"], document["doc_topics"],
                                document["topic_changes"], document["topic_counts"])
    model.add_document(c_doc)

model.sample(1000)

with open(args.output_state, 'w') as state_out:
    print('#doc source pos typeindex type topic', file=state_out)
    print('#alpha : {}'.format(' '.join([str(doc_smoothing) for _ in range(num_topics)])), file=state_out)
    print('#beta : {}'.format(str(word_smoothing)), file=state_out)

    for doc_i, document in enumerate(model.documents):
        for token_j in range(len(document.doc_tokens)):
            print(' '.join([str(doc_i), str(document.doc_id), str(token_j), str(document.doc_tokens[token_j]),
                            str(vocabulary[document.doc_tokens[token_j]]), str(document.doc_topics[token_j])]),
                  file=state_out)

with open(args.output_topic_keys, 'w') as key_out:
    model.print_all_topics(out=key_out)

# sample(1000)
# topicmodel.sample(10, documents, word_topics, topic_totals, doc_smoothing, word_smoothing,
# smoothing_times_vocab_size, num_topics)
# print_all_topics()
# with open("state.txt", "w") as writer:
#    write_state(writer)
