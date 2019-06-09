import csv
import data_process


def cutoff_prob(input_path: str, output_path: str):
    with open(input_path, 'r', newline='') as input_file:
        csv_reader = csv.reader(input_file)
        csv_reader.__next__()

        with open(output_path, 'w', newline='') as output_file:
            csv_writer = csv.writer(output_file)
            csv_writer.writerow(["id"] + data_process.TAGS)

            for row in csv_reader:
                row_id = row[0]
                probs = list(map(float, row[1:]))

                csv_writer.writerow([row_id, ] + probs)


if __name__ == '__main__':
    cutoff_prob(input_path="submission2.csv", output_path="submission2_post.csv")
