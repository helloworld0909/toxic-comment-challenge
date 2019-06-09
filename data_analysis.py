import collections
import matplotlib.pyplot as plt

import data_loader
import data_process
import util


def label_distribution(df, name="train"):
    counter = {tag: collections.defaultdict(int) for tag in data_process.TAGS}

    for idx, row in df.iterrows():
        for tag in data_process.TAGS:
            counter[tag][row[tag]] += 1

    positive, negative = [], []
    for tag in data_process.TAGS:
        dist = counter[tag]
        for label, count in dist.items():
            if label == 0:
                negative.append(count)
            else:
                positive.append(count)

    ind = range(len(data_process.TAGS))
    plt.bar(ind, negative, 0.35, label="negative")
    plt.bar(ind, positive, 0.35, label="positive", bottom=negative)
    plt.title("label distribution {}".format(name))
    plt.ylabel("count")
    plt.xticks(ind, data_process.TAGS)
    plt.legend()
    plt.savefig("label_distribution_{}.png".format(name))
    plt.close()


def sentence_length_distribution():
    freq = 10
    x_tr, y_tr, x_va, y_va, dic, x_te, id_list = util.create_or_load_data(freq_threshold=freq)
    counter_tr, counter_va = [collections.defaultdict(lambda: collections.defaultdict(int))] * 2
    for x, y in zip(x_tr, y_tr):
        for i in range(len(y)):
            counter_tr["{}_{}".format(i, y[i])][len(x)] += 1

    for x, y in zip(x_va, y_va):
        for i in range(len(y)):
            counter_va["{}_{}".format(i, y[i])][len(x)] += 1

    lengths_tr = collections.defaultdict(list)
    counts_tr = collections.defaultdict(list)
    for k in counter_tr:

        for length, c in sorted(counter_tr[k].items()):
            lengths_tr[k].append(length)
            counts_tr[k].append(c)

    bins = range(0, 2000, 20)
    plt.hist(lengths_tr["0_0"], bins=bins, weights=counts_tr["0_0"], label="non-toxic train")
    plt.hist(lengths_tr["0_1"], bins=bins, weights=counts_tr["0_1"], label="toxic train")
    plt.title("sentence length distribution of train data")
    plt.ylabel("count")
    plt.xlabel("length")
    plt.yscale("log")
    plt.legend()
    plt.savefig("sentence_length_distribution_{}_train.png".format(freq))
    plt.close()


if __name__ == '__main__':
    train_df, valid_df = data_loader.load_train_data("./resources/train.csv")

    label_distribution(train_df, name="train")
    label_distribution(valid_df, name="valid")
    # sentence_length_distribution()
