from flask import Flask, render_template, jsonify, request

# 1. ต้องมีบรรทัดนี้เพื่อสร้างตัวแปร 'app' ก่อน
app = Flask(__name__)

# 2. ตัวแปรเก็บข้อมูลขยะ (สมมติค่าเริ่มต้นเป็น 0)
waste_counts = {
    "bottle": 0,
    "glass": 0,
    "can": 0,
    "other": 0
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def get_data():
    return jsonify(waste_counts)

# 3. ส่วนของ Reset ที่คุณเพิ่งเพิ่มเข้าไป
@app.route('/reset', methods=['POST'])
def reset_data():
    global waste_counts
    waste_counts = {
        "bottle": 0, "glass": 0, "can": 0, "other": 0
    }
    return jsonify({"status": "success"})

# 4. ส่วนสำหรับรันโปรแกรม
if __name__ == '__main__':
    app.run(debug=True)