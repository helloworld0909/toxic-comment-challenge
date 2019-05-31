import csv
import numpy as np
from sklearn.metrics import roc_auc_score
import data_process


def auc(Y, Y_pred):
    auc_sum = 0
    for label, y_pred in zip(Y, Y_pred):
        y_hat = np.zeros(len(y_pred))
        y_hat[label] = 1
        auc_sum += roc_auc_score(y_hat, y_pred)
    return auc_sum / len(Y)


def submission(pred_list, id_list, output_path="submission.csv"):
    with open(output_path, 'w', newline='') as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(["id"] + data_process.TAGS)

        for idx in range(len(id_list)):
            row = [id_list[idx]]
            for j in range(len(data_process.TAGS)):
                positive_prob = pred_list[j][idx][1]
                row.append(positive_prob)
            csv_writer.writerow(row)