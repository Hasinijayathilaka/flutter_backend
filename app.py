from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

# Helper: Convert numpy types
def convert_np_types(obj):
    if isinstance(obj, dict):
        return {k: convert_np_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_types(i) for i in obj]
    elif isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj

# Main parsing logic
def parse_sms(sms_text):
    lower = sms_text.lower()
    result = {
        "original_text": sms_text,
        "is_bank_sms": False,
        "is_reminder": False,
        "type": None,
        "amount": None,
        "balance": None,
        "account_or_card": None,
        "entities": [],
        "utility": None,
        "due_date": None,
        "is_active_reminder": False,
        "parsed_at": datetime.now().isoformat()
    }

    if ("bill" in lower and ("due on" in lower or "by" in lower or "before" in lower)) or \
       ("outstanding" in lower and any(u in lower for u in ["electri", "water", "bill"])) or \
       "total due" in lower or any(t in lower for t in ["reading", "units", "ssc levy", "monthly bill"]):
        result["is_reminder"] = True

    bank_kw = ["a/c", "card", "credited", "debited", "transaction", "balance", "avl bal"]
    if any(k in lower for k in bank_kw) and not result["is_reminder"]:
        result["is_bank_sms"] = True

    amt = re.search(r"(?:lkr|rs\.?)\s*([\d,]+\.\d{1,2})", sms_text, re.IGNORECASE)
    if amt:
        result["amount"] = float(amt.group(1).replace(",", ""))

    due_amt = re.search(r"(?:outstanding|total due)\s*[:\-]?\s*(?:lkr|rs\.?)?\s*([\d,]+\.\d{1,2})", sms_text, re.IGNORECASE)
    if due_amt:
        result["amount"] = float(due_amt.group(1).replace(",", ""))

    bal = re.search(r"(?:balance(?: available| is)?|avl bal)[\s:\-]*(?:lkr|rs\.?)\s*([\d,]+\.\d{1,2})", sms_text, re.IGNORECASE)
    if bal:
        result["balance"] = float(bal.group(1).replace(",", ""))

    acc = re.search(r"a/c\s?no\.?\s?\*+(\d+)", sms_text, re.IGNORECASE)
    if acc:
        result["account_or_card"] = f"A/C No ****{acc.group(1)}"
    else:
        crd = re.search(r"card\s?(\d+)", sms_text, re.IGNORECASE)
        if crd:
            result["account_or_card"] = f"Card {crd.group(1)}"

    if result["is_bank_sms"]:
        if "credited" in lower or "deposit" in lower:
            result["type"] = "income"
        elif "debited" in lower or "spent" in lower or "transaction" in lower:
            result["type"] = "expense"

    if result["is_reminder"]:
        if "electric" in lower or any(k in lower for k in ["reading", "units", "ssc levy", "monthly bill"]):
            result["utility"] = "Electricity"
        elif "water" in lower:
            result["utility"] = "Water"
        elif any(t in lower for t in ["dialog", "mobitel", "telecom", "network", "data pack"]):
            result["utility"] = "Telecom"

        date = re.search(r"(?:due on|by|before)\s?(\d{4}-\d{2}-\d{2})", sms_text, re.IGNORECASE)
        if not date:
            date = re.search(r"(?:due on|by|before)\s?(\d{2}-\d{2}-\d{4})", sms_text, re.IGNORECASE)

        if date:
            result["due_date"] = date.group(1)
            try:
                try:
                    due = datetime.strptime(result["due_date"], "%Y-%m-%d")
                except ValueError:
                    due = datetime.strptime(result["due_date"], "%d-%m-%Y")
                result["is_active_reminder"] = due >= datetime.now()
            except:
                result["is_active_reminder"] = True

    return result

def is_same_month(date_str):
    try:
        dt = datetime.fromisoformat(date_str)
        now = datetime.now()
        return dt.year == now.year and dt.month == now.month
    except:
        return False

@app.route('/')
def index():
    return "✅ Flask SMS Parser API is running!"

@app.route('/parse_sms', methods=['POST'])
def parse_single():
    data = request.get_json(force=True)
    sms = data.get('sms_text', "")
    print(f"🔹 /parse_sms received: {sms[:60]}...")
    return jsonify(parse_sms(sms)) if sms else (jsonify({"error": "Missing sms_text"}), 400)

@app.route('/parse_sms_bulk', methods=['POST'])
def parse_bulk():
    data = request.get_json(force=True)
    sms_list = data.get('sms_list', [])
    print(f"🔸 /parse_sms_bulk received {len(sms_list)} messages")

    bank = {}
    reminders = []

    for sms in sms_list:
        p = parse_sms(sms)
        if p["is_bank_sms"]:
            acc = p["account_or_card"] or "unknown"
            bank.setdefault(acc, {
                "total_income": 0,
                "total_expense": 0,
                "monthly_income": 0,
                "monthly_expense": 0,
                "latest_balance": None,
                "messages": []
            })

            amount = p.get("amount", 0) or 0
            if p["type"] == "income":
                bank[acc]["total_income"] += amount
                if is_same_month(p.get("parsed_at", "")):
                    bank[acc]["monthly_income"] += amount
            elif p["type"] == "expense":
                bank[acc]["total_expense"] += amount
                if is_same_month(p.get("parsed_at", "")):
                    bank[acc]["monthly_expense"] += amount

            if p["balance"] is not None:
                bank[acc]["latest_balance"] = p["balance"]
            bank[acc]["messages"].append(p)

        elif p["is_reminder"] and p.get("is_active_reminder"):
            reminders.append({
                "utility": p["utility"],
                "amount": p["amount"],
                "due_date": p["due_date"],
                "original_text": p["original_text"]
            })

    return jsonify({
        "bank_sms": convert_np_types(bank),
        "reminders": reminders
    })

# ✅ IMPORTANT: Cloud-friendly port binding
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
