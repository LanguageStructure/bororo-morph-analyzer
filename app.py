from flask import Flask, render_template, request, jsonify
import joblib
import os
from os import listdir
from os.path import join

app = Flask(__name__, template_folder="templates")

# === Load CRF model for analyzer (old version) ===
model = joblib.load("/home/Boeenomoto/Morfologia/bororo_morph_model_lr.joblib")
vectorizer = joblib.load("/home/Boeenomoto/Morfologia/bororo_vectorizer_lr.joblib")
mlb = joblib.load("/home/Boeenomoto/Morfologia/bororo_labelbinarizer_lr.joblib")

# === Load slot-based CRFs ===
crf_slot_models = {}
slot_path = "/home/Boeenomoto/Morfologia/crf_slots/"
if os.path.exists(slot_path):
    for filename in listdir(slot_path):
        if filename.endswith(".joblib"):
            slot = filename.replace("crf_", "").replace(".joblib", "")
            crf_slot_models[slot] = joblib.load(join(slot_path, filename))

# === Feature extractor ===
def extract_features(word):
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

# === Routes ===
@app.route("/")
def index():
    return render_template("main.html")

@app.route("/Morfologia")
def morfologia_page():
    return render_template("index.html")

@app.route("/analisar")
def analisar():
    palavra = request.args.get("palavra", "")
    if not palavra:
        return jsonify({"erro": "Parâmetro 'palavra' ausente"}), 400
    try:
        vec = vectorizer.transform([palavra])
        pred = model.predict(vec)
        feats = mlb.inverse_transform(pred)
        return jsonify({
            "palavra": palavra,
            "morfemas": feats[0] if feats else []
        })
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

@app.route("/analisar_slots")
def analisar_slots():
    palavra = request.args.get("palavra", "")
    if not palavra:
        return jsonify({"erro": "Parâmetro 'palavra' ausente"}), 400

    try:
        features = [extract_features(palavra)]
        print(f"PALAVRA: {palavra}")
        results = {}

        for slot, model in crf_slot_models.items():
            try:
                pred = model.predict([features])[0][0]
                print(f"{slot} → {pred}")
                results[slot] = pred
            except Exception as e:
                print(f"Erro em {slot}: {e}")
                results[slot] = "?"

        return jsonify({
            "palavra": palavra,
            "morfemas": results
        })

    except Exception as e:
        return jsonify({"erro": str(e)}), 500
