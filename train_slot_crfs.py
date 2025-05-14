from conllu import parse_incr
import sklearn_crfsuite
from sklearn.model_selection import train_test_split
from sklearn_crfsuite.metrics import flat_classification_report
import os, joblib
from collections import defaultdict

# === SETTINGS ===
train_path = "/home/Boeenomoto/Morfologia/bor_bdt-ud-train.conllu"
test_path  = "/home/Boeenomoto/Morfologia/bor_bdt-ud-test.conllu"
MIN_EXAMPLES = 10  # Minimum number of examples to train a CRF

# === Feature extractor ===
def word2features(word):
    word = word.lower()
    return {
        "word": word,
        "prefix1": word[:1],
        "prefix2": word[:2],
        "prefix3": word[:3],
        "suffix1": word[-1:],
        "suffix2": word[-2:],
        "suffix3": word[-3:],
        "suffix4": word[-4:] if len(word) >= 4 else "",
        "suffix5": word[-5:] if len(word) >= 5 else "",
        "suffix6": word[-6:] if len(word) >= 6 else "",
        "len": len(word),
        "is_title": word.istitle(),
        "is_digit": word.isdigit()
    }

# === Parse CONLLU and split into separate features ===
def load_by_feature(conllu_path):
    feature_data = defaultdict(lambda: {"X": [], "y": []})
    with open(conllu_path, "r", encoding="utf-8") as f:
        for sentence in parse_incr(f):
            feats_by_feature = defaultdict(list)
            labels_by_feature = defaultdict(list)
            for token in sentence:
                form = token.get("form")
                feats = token.get("feats")
                if form and feats:
                    features = word2features(form)
                    for k in feats:
                        feats_by_feature[k].append(features)
                        labels_by_feature[k].append(feats[k])
            for feat_key in feats_by_feature:
                feature_data[feat_key]["X"].append(feats_by_feature[feat_key])
                feature_data[feat_key]["y"].append(labels_by_feature[feat_key])
    return feature_data

# === Load both train + test ===
data_train = load_by_feature(train_path)
data_test  = load_by_feature(test_path)

# === Merge all features across files ===
all_features = set(data_train.keys()).union(data_test.keys())

for key in sorted(all_features):
    X = data_train.get(key, {"X": []})["X"] + data_test.get(key, {"X": []})["X"]
    y = data_train.get(key, {"y": []})["y"] + data_test.get(key, {"y": []})["y"]

    flat_count = sum(len(seq) for seq in y)
    if flat_count < MIN_EXAMPLES:
        print(f"⏩ Skipping {key}: only {flat_count} examples")
        continue

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\n🔧 Training CRF for feature: {key} ({flat_count} tokens)")
    crf = sklearn_crfsuite.CRF(algorithm="ap", max_iterations=50)
    crf.fit(X_train, y_train)

    # Save
    os.makedirs("/home/Boeenomoto/Morfologia/crf_slots", exist_ok=True)
    joblib.dump(crf, f"/home/Boeenomoto/Morfologia/crf_slots/crf_{key}.joblib")

    # Evaluate
    y_pred = crf.predict(X_test)
    print(f"📊 Evaluation for {key}:")
    print(flat_classification_report(y_test, y_pred, digits=3))
