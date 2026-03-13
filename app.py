@app.route('/reset', methods=['POST'])
def reset_data():
    global waste_counts
    # ตั้งค่ากลับเป็น 0 ทั้งหมด
    waste_counts = {
        "bottle": 0,
        "glass": 0,
        "can": 0,
        "other": 0
    }
    return jsonify({"status": "success", "message": "Counters reseted"})