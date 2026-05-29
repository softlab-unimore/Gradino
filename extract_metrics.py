import pandas as pd
import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument('--filepath', type=str, required=True, help='CSV file path')
args = vars(parser.parse_args())

path = args['filepath']
base_dir = os.path.dirname(path)
path_to_save = os.path.join(base_dir, "preds_with_match.csv")
score_path = os.path.join(base_dir, "score.csv")

df = pd.read_csv(path)

def extract_first_number(series):
    return pd.to_numeric(
        series.astype(str)
        .str.extract(r'([-+]?(?:\d[\d,]*\.?\d*|\.\d+))')[0]
        .str.replace(",", "", regex=False),
        errors="coerce"
    )

def normalize_text(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r'[^a-z0-9]+', '', regex=True)
    )

pred_num = extract_first_number(df["Prediction"])
label_num = extract_first_number(df["Label"])

numeric_mode = label_num.notna().sum() > 0 and label_num.notna().sum() == len(df)

if numeric_mode:
    df["Match"] = (pred_num - label_num).abs() < 0.005
else:
    pred_text = normalize_text(df["Prediction"])
    label_text = normalize_text(df["Label"])

    def to_binary_label(s):
        s = str(s)
        if s.startswith("yes"):
            return "yes"
        if s.startswith("no"):
            return "no"
        return s

    pred_text = pred_text.apply(to_binary_label)
    label_text = label_text.apply(to_binary_label)

    df["Match"] = pred_text == label_text

df.to_csv(path_to_save, index=False)

accuracy = df["Match"].mean()
score_df = pd.DataFrame({"Accuracy": [accuracy]})
score_df.to_csv(score_path, index=False)

print(f"Saved predictions to: {path_to_save}")
print(f"Saved score to: {score_path}")
print(score_df)
