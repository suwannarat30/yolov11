from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

data = {
    "bottle": 0,
    "glass": 0,
    "can": 0,
    "other": 0
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/type/<name>")
def type_page(name):
    return render_template("type.html", waste=name)

@app.route("/data")
def get_data():
    return jsonify(data)

@app.route("/add", methods=["POST"])
def add():

    waste = request.json["type"].lower()

    # แปลงชื่อจาก YOLO
    if "bottle" in waste:
        data["bottle"] += 1

    elif "glass" in waste:
        data["glass"] += 1

    elif "can" in waste:
        data["can"] += 1

    else:
        data["other"] += 1

    print("Update:", data)

    return jsonify({"status":"ok"})


app.run(host="0.0.0.0", port=5000)