import collections
import matplotlib.pyplot as plt

import data_loader
import data_process


def label_distribution(df):
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
    plt.title("label distribution")
    plt.ylabel("count")
    plt.xticks(ind, data_process.TAGS)
    plt.legend()
    plt.show()


def sentence_length_distribution():
    x_tr, _, x_va, _ = data_process.load_processed_train_data("./resources/train.csv")
    counter_tr, counter_va = [collections.defaultdict(int)] * 2
    for x in x_tr:
        counter_tr[len(x)] += 1
    for x in x_va:
        counter_va[len(x)] += 1

    plt.hist(counter_tr.values(), bins=100, label="train")
    plt.title("sentence length distribution of train data")
    plt.ylabel("count")
    plt.xlabel("length")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train_df, valid_df = data_loader.load_train_data("./resources/train.csv")

    label_distribution(train_df)
    label_distribution(valid_df)
    sentence_length_distribution()
