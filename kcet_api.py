from flask import Flask, request, jsonify
import pandas as pd
import pickle
from pathlib import Path
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model_path = Path("kcet_model.pkl")
if not model_path.exists():
    raise FileNotFoundError("Model file not found! Train and save 'model.pkl' first.")

with open(model_path, "rb") as file:
    model = pickle.load(file)

#
def load_college_data():
    file_path = Path("kcet_colleges.csv")
    if not file_path.exists():
        raise FileNotFoundError("College data file not found! Ensure 'data.csv' exists.")
    df = pd.read_csv(file_path)
    df.dropna(subset=["CETCode", "College", "Location", "Branch"], inplace=True)
    return df

@app.route("/predict_kcet", methods=["POST"])
def predict_kcet():
    """Predict KCET rank from PCM and KCET scores, then fetch eligible colleges."""
    try:
        data = request.json
        phy = data.get("phy")
        chem = data.get("chem")
        math = data.get("math")
        kcet = data.get("kcet")
        category = data.get("category")
        branches = data.get("branches", [])

        if None in [phy, chem, math, kcet, category]:
            return jsonify({"error": "Please provide physics, chemistry, math, kcet marks, and category"}), 400

        
        try:
            phy, chem, math, kcet = map(int, [phy, chem, math, kcet])
        except ValueError:
            return jsonify({"error": "Invalid input. PCM and KCET should be integers"}), 400

        
        pcm_total = phy + chem + math
        input_data = pd.DataFrame([[pcm_total, kcet]], columns=['pcm_total', 'kcet'])
        predicted_rank = int(model.predict(input_data)[0])

        
        df = load_college_data()
        if category not in df.columns:
            return jsonify({"error": "Invalid category"}), 400

        
        df[category] = pd.to_numeric(df[category], errors="coerce").fillna(0).astype(int)

    
        eligible_colleges = df[df[category] > predicted_rank]

        
        if branches:
            eligible_colleges = eligible_colleges[eligible_colleges["Branch"].isin(branches)]

        if eligible_colleges.empty:
            return jsonify({
                "predicted_rank": predicted_rank,
                "message": "Sorry, no colleges available."
            })

        
        output_df = eligible_colleges[["CETCode", "College", "Branch", category]].rename(columns={category: "Cutoff"})
        output_df = output_df.sort_values(by="Cutoff")

        return jsonify({
            "phy": phy,
            "chem": chem,
            "math": math,
            "kcet": kcet,
            "predicted_rank": predicted_rank,
            "eligible_colleges": output_df.to_dict(orient="records")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

#test data
# {
#   "phy": 80,
#   "chem": 75,
#   "math": 50,
#   "kcet": 85,
#   "category": "GM",
#   "branches": ["CS", "AI"]
# }
