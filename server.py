from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# กำหนดความจุของแต่ละถัง (ปรับได้ที่นี่)
LIMITS = {
    "bottle": 10,
    "glass":  10,
    "can":    10,
    "other":  10
}

data = {
    "bottle": 0,
    "glass":  0,
    "can":    0,
    "other":  0
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/data")
def get_data():
    return jsonify({"counts": data, "limits": LIMITS})

@app.route("/add", methods=["POST"])
def add():
    waste = request.json["type"].lower()

    if "bottle" in waste:
        key = "bottle"
    elif "glass" in waste:
        key = "glass"
    elif "can" in waste:
        key = "can"
    else:
        key = "other"

    # ไม่บวกเกิน limit
    if data[key] < LIMITS[key]:
        data[key] += 1

    print("Update:", data)
    return jsonify({"status": "ok", "counts": data, "limits": LIMITS})

@app.route("/reset", methods=["POST"])
def reset():
    for key in data:
        data[key] = 0
    print("Reset:", data)
    return jsonify({"status": "ok", "counts": data, "limits": LIMITS})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)