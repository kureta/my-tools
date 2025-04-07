import marimo

__generated_with = "0.12.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    from enum import Enum

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    from scipy.stats import entropy
    import librosa

    mo.md("# Constrained chords")
    return Enum, Path, entropy, librosa, mo, np, pl, plt


@app.cell
def _():
    PIANO_OFFSET = 21
    PIANO_HIGHEST = 109
    PIANO_NKEYS = 88
    return PIANO_HIGHEST, PIANO_NKEYS, PIANO_OFFSET


@app.cell
def _(Enum):
    class Interval(Enum):
        P1 = 0
        MIN2 = 1
        MAJ2 = 2
        MIN3 = 3
        MAJ3 = 4
        P4 = 5
        A4D5 = 6
        P5 = 7
        MIN6 = 8
        MAJ6 = 9
        MIN7 = 10
        MAJ7 = 11
        P8 = 12
    return (Interval,)


@app.cell
def _(entropy, np, pl):
    def contains_subsequence_np(array, subsequence):
        array = np.array(array)
        subsequence = np.array(subsequence)

        # Create a sliding window view that matches subsequence length
        sub_len = len(subsequence)
        windows = np.lib.stride_tricks.sliding_window_view(array, sub_len)

        # Check each window against the subsequence
        return np.any(np.all(windows == subsequence, axis=1))


    def generate_possible_sets(intervals, start=0, max_value=88, target_size=12):
        def is_valid_set(candidate_set, used_pitch_classes):
            # Check all elements are within bounds and distinct modulo 12
            return len(candidate_set) == len(set(candidate_set)) and len(
                used_pitch_classes
            ) == len(candidate_set)

        def backtrack(current, candidate_set, used_pitch_classes):
            if len(candidate_set) == target_size:
                # Terminal condition: a valid set is complete
                results.append(list(candidate_set))
                return
            if current > max_value:
                # Stop if the current number exceeds the max_value
                return

            for interval in intervals:
                next_number = current + interval
                mod_value = next_number % 12

                if next_number > max_value or mod_value in used_pitch_classes:
                    # Skip if out of bounds or if modulo constraint is violated
                    continue

                # Include the next number and its modulo
                candidate_set.append(next_number)
                used_pitch_classes.add(mod_value)

                backtrack(next_number, candidate_set, used_pitch_classes)

                # Backtrack: remove the last number and its modulo
                candidate_set.pop()
                used_pitch_classes.remove(mod_value)

        results = []
        for start_num in range(max_value + 1):
            backtrack(start_num, [start_num], {start_num % 12})

        return results


    def calculate_entropy(sequence):
        return entropy([sequence.count(value) for value in set(sequence)])


    def extract_n_tuples(data, n):
        return [tuple(data[i : i + n]) for i in range(len(data) - n + 1)]


    # Add all necessary calculated properties here
    def create_lazyframe_from_sets(sets):
        df = pl.DataFrame(
            {
                "set": [np.array(s) for s in sets],
                "initial_number": [s[0] for s in sets],
                "final_number": [s[-1] for s in sets],
                "span": [s[-1] - s[0] for s in sets],
                "interval_sequence": [
                    np.array([s[i + 1] - s[i] for i in range(len(s) - 1)])
                    for s in sets
                ],
            }
        )

        df = df.with_columns(
            [
                pl.col("interval_sequence")
                .map_elements(calculate_entropy, return_dtype=float)
                .alias("interval_entropy"),
                pl.col("interval_sequence")
                .map_elements(
                    lambda x: calculate_entropy(extract_n_tuples(x, 2)),
                    return_dtype=float,
                )
                .alias(f"bigram_entropy"),
                pl.col("interval_sequence")
                .map_elements(
                    lambda x: calculate_entropy(extract_n_tuples(x, 3)),
                    return_dtype=float,
                )
                .alias(f"trigram_entropy"),
                pl.col("interval_sequence")
                .map_elements(
                    lambda x: calculate_entropy(extract_n_tuples(x, 4)),
                    return_dtype=float,
                )
                .alias(f"quadrigram_entropy"),
            ]
        )

        return df.lazy()
    return (
        calculate_entropy,
        contains_subsequence_np,
        create_lazyframe_from_sets,
        extract_n_tuples,
        generate_possible_sets,
    )


@app.cell
def _(generate_possible_sets):
    intervals = [7, 11]
    sets = generate_possible_sets(intervals)
    # While it's not practical to print all the results for large outputs, a sample verification can be done
    print("Number of valid sets found:", len(sets))
    # Feel free to inspect sets or write additional tests to check some result samples
    return intervals, sets


@app.cell
def _(contains_subsequence_np, create_lazyframe_from_sets, mo, pl, sets):
    subseqs = [[1, 1], [1, 2], [2, 1], [2, 2]]
    df = create_lazyframe_from_sets(sets)
    df = (
        df.filter((pl.col("interval_entropy") > 0.0))
        .filter((pl.col("initial_number") == 0))
        .filter(
            pl.col("interval_sequence").map_elements(
                lambda seq: not any(
                    contains_subsequence_np(seq, subseq) for subseq in subseqs
                ),
                return_dtype=bool,
            )
        )
        .sort(
            [
                "interval_entropy",
                "bigram_entropy",
                "trigram_entropy",
                "quadrigram_entropy",
            ],
            descending=True,
        )
    )


    df_min_max = df.select(
        [
            pl.col("interval_entropy").min().alias("min_entropy"),
            pl.col("interval_entropy").max().alias("max_entropy"),
            pl.col("bigram_entropy").min().alias("min_bigram_entropy"),
            pl.col("bigram_entropy").max().alias("max_bigram_entropy"),
            pl.col("trigram_entropy").min().alias("min_trigram_entropy"),
            pl.col("trigram_entropy").max().alias("max_trigram_entropy"),
            pl.col("quadrigram_entropy").min().alias("min_quadrigram_entropy"),
            pl.col("quadrigram_entropy").max().alias("max_quadrigram_entropy"),
            pl.col("span").min().alias("min_span"),
            pl.col("span").max().alias("max_span"),
        ]
    )

    mo.vstack(
        [
            df.collect(),
            df_min_max.collect(),
            all(len(set([p % 12 for p in s])) == 12 for s in sets),
        ]
    )
    return df, df_min_max, subseqs


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
