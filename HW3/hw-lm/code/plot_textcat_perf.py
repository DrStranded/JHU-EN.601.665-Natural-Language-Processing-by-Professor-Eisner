import numpy as np
import matplotlib.pyplot as plt

def binned_accuracy(file_lens, outcomes, n_bins=50):
    file_lens = np.array(file_lens)
    outcomes = np.array(outcomes)

    bins = np.linspace(file_lens.min(), file_lens.max(), n_bins + 1)
    bin_indices = np.digitize(file_lens, bins, right=True)

    bin_centers = []
    accuracies = []

    for i in range(1, n_bins + 1):
        mask = (bin_indices == i)
        if mask.sum() == 0:
            continue
        acc = outcomes[mask].mean()
        center = (bins[i - 1] + bins[i]) / 2
        bin_centers.append(center)
        accuracies.append(acc)

    return np.array(bin_centers), np.array(accuracies)

pred_gen_file = "textcat_dev_gen.txt"
pred_spam_file = "textcat_dev_spam.txt"

file_lens = []
outcomes = []

with open(pred_gen_file) as file:
    for line in file:
        tokens = line.split()
        if len(tokens) != 2:
            continue

        pred_model = tokens[0].split("_")[0]
        data_file = tokens[1]

        correct = (1 if pred_model == "gen" else 0)
        file_len = int(data_file.split(".")[-3])
        file_lens.append(file_len)
        outcomes.append(correct)

with open(pred_spam_file) as file:
    for line in file:
        tokens = line.split()
        if len(tokens) != 2:
            continue

        pred_model = tokens[0].split("_")[0]
        data_file = tokens[1]

        correct = (1 if pred_model == "spam" else 0)
        file_len = int(data_file.split(".")[-3])
        file_lens.append(file_len)
        outcomes.append(correct)

bin_centers, accuracies = binned_accuracy(file_lens, outcomes, 100)
plt.scatter(bin_centers, accuracies)
plt.show()