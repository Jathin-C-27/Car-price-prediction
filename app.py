import os
import pickle
from datetime import datetime

import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "model.pkl not found. Run the training notebook and put model.pkl "
        "in the same folder as app.py."
    )

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Encoding maps: must match the training notebook exactly
BRANDS = [
    "Maruti", "Skoda", "Honda", "Hyundai", "Toyota", "Ford", "Renault",
    "Mahindra", "Tata", "Chevrolet", "Datsun", "Jeep", "Mercedes-Benz",
    "Mitsubishi", "Audi", "Volkswagen", "BMW", "Nissan", "Lexus",
    "Jaguar", "Land", "MG", "Volvo", "Daewoo", "Kia", "Fiat", "Force",
    "Ambassador", "Ashok", "Isuzu", "Opel",
]
BRAND_MAP = {brand: i + 1 for i, brand in enumerate(BRANDS)}
FUEL_MAP = {"Diesel": 1, "Petrol": 2, "LPG": 3, "CNG": 4}
SELLER_MAP = {"Individual": 1, "Dealer": 2, "Trustmark Dealer": 3}
TRANSMISSION_MAP = {"Manual": 1, "Automatic": 2}
OWNER_MAP = {
    "First Owner": 1, "Second Owner": 2, "Third Owner": 3,
    "Fourth & Above Owner": 4, "Test Drive Car": 5,
}

# Column order the model was trained with
FEATURES = [
    "name", "year", "km_driven", "fuel", "seller_type", "transmission",
    "owner", "mileage", "engine", "max_power", "seats",
]

DEFAULTS = {
    "name": "Maruti", "year": "2018", "km_driven": "40000", "fuel": "Petrol",
    "seller_type": "Individual", "transmission": "Manual",
    "owner": "First Owner", "mileage": "18", "engine": "1200",
    "max_power": "85", "seats": "5",
}

CHOICES = {
    "name": BRANDS,
    "fuel": list(FUEL_MAP),
    "seller_type": list(SELLER_MAP),
    "transmission": list(TRANSMISSION_MAP),
    "owner": list(OWNER_MAP),
}

CURRENT_YEAR = datetime.now().year

# (label, minimum, maximum, type)
NUMBER_RULES = {
    "year": ("Year", 1994, CURRENT_YEAR, int),
    "km_driven": ("Kilometres driven", 0, 1_000_000, int),
    "mileage": ("Mileage (kmpl)", 1, 50, float),
    "engine": ("Engine (cc)", 500, 6000, float),
    "max_power": ("Max power (bhp)", 20, 700, float),
    "seats": ("Seats", 2, 14, int),
}


def format_inr(amount):
    """Format a number with Indian digit grouping, e.g. 1234567 -> 12,34,567."""
    amount = int(round(amount))
    digits = str(abs(amount))
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        digits = ",".join(groups + [tail])
    return ("-" if amount < 0 else "") + digits


def validate(values):
    """Return (clean_values, errors). Values may come from a form or JSON."""
    clean, errors = {}, {}

    for field, options in CHOICES.items():
        value = str(values.get(field, "")).strip()
        if value not in options:
            errors[field] = "Choose one of the listed options."
        else:
            clean[field] = value

    for field, (label, low, high, kind) in NUMBER_RULES.items():
        raw = str(values.get(field, "")).strip()
        try:
            number = kind(float(raw)) if kind is int else float(raw)
        except ValueError:
            errors[field] = f"{label} must be a number."
            continue
        if not low <= number <= high:
            errors[field] = f"{label} must be between {low} and {high}."
        else:
            clean[field] = number

    return clean, errors


def predict_price(clean):
    row = [[
        BRAND_MAP[clean["name"]], clean["year"], clean["km_driven"],
        FUEL_MAP[clean["fuel"]], SELLER_MAP[clean["seller_type"]],
        TRANSMISSION_MAP[clean["transmission"]], OWNER_MAP[clean["owner"]],
        clean["mileage"], clean["engine"], clean["max_power"], clean["seats"],
    ]]
    frame = pd.DataFrame(row, columns=FEATURES)
    price = float(model.predict(frame)[0])
    return max(price, 0.0)  # a linear model can go below zero for odd inputs


@app.route("/", methods=["GET", "POST"])
def index():
    values = dict(DEFAULTS)
    errors, price, summary = {}, None, None

    if request.method == "POST":
        values.update({k: request.form.get(k, "") for k in DEFAULTS})
        clean, errors = validate(values)
        if not errors:
            price = format_inr(predict_price(clean))
            summary = clean

    return render_template(
        "index.html", values=values, errors=errors, price=price,
        summary=summary, choices=CHOICES, current_year=CURRENT_YEAR,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send a JSON object with the car details."}), 400

    clean, errors = validate(payload)
    if errors:
        return jsonify({"errors": errors}), 400

    price = predict_price(clean)
    return jsonify({"predicted_price": round(price), "currency": "INR"})


if __name__ == "__main__":
    app.run(debug=True)
