''' to read .npz file'''
import numpy as np


def load_dataset(path="radar_dataset.npz"):

    data = np.load(path)

    X = data["X"]
    y = data["y"]

    return X, y


def dataset_summary(X, y):

    print("=================================")
    print("Dataset Summary")
    print("=================================")

    print("Samples:", X.shape[0])
    print("Features:", X.shape[1])

    print("\nLabel distribution:")

    unique, counts = np.unique(y, return_counts=True)

    label_map = {
        0: "empty",
        1: "aircraft",
        2: "stealth"
    }

    for u, c in zip(unique, counts):
        print(f"{label_map.get(int(u), u)} : {c}")

    print("\nFeature stats:")

    print("Mean:", np.mean(X))
    print("Std :", np.std(X))
    print("Min :", np.min(X))
    print("Max :", np.max(X))

    print("\nNaN check:", np.isnan(X).any())
    print("Inf check:", np.isinf(X).any())


def inspect_samples(X, y, n=5):

    print("\n=================================")
    print("Sample rows")
    print("=================================")

    for i in range(min(n, len(X))):
        print(f"\nSample {i}")
        print("Label:", y[i])
        print("Features:", X[i])


def main():

    X, y = load_dataset()

    dataset_summary(X, y)

    inspect_samples(X, y, n=5)


if __name__ == "__main__":
    main()