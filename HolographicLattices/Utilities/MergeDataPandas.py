import pandas as pd

import argparse
parser = argparse.ArgumentParser(prog="MergeCSV", usage="Combine multiple files together by concatenating them. If there are non-matching columns, new entries with value NaN will be created (see pandas.concat documentation)", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input', type=str, nargs="+", help="Files to combine together")
parser.add_argument('--output', type=str, default="LatestMerge.tsv", help="File where to output the result")
parser.add_argument('--sort', type=str, default="omega", help = "Column name to sort by (ascending)")
parser.add_argument('--unique_column', action="store_true", default=False, help = "Make the sorted column unique (up to floating point)")

parsed,_ = parser.parse_known_args()

merged_df = pd.DataFrame()
for f in parsed.files:
    data = pd.read_csv(f, sep="\t")
    merged_df = pd.concat((merged_df, data))

merged_df.sort_values(by=parsed.sort, inplace=True)

if parsed.unique_column:
    print(f"Dropping duplicates, keeping first in column {parsed.sort}")
    merged_df.drop_duplicates(subset=[parsed.sort],inplace=True)

print(f"Saving output to {parsed.output}")
merged_df.to_csv(parsed.output, sep="\t", index=False, header=True)
