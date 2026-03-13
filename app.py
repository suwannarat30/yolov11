from flask import Flask, render_template, jsonify, request
import os # เพิ่มบรรทัดนี้เพื่อดึงค่าจากระบบ

app = Flask(__name__)

# ตัวแปรเก็บข้อมูลขยะ
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

@app.route('/reset', methods=['POST'])
def reset_data():
    global waste_counts
    waste_counts = {
        "bottle": 0, "glass": 0, "can": 0, "other": 0
    }
    return jsonify({"status": "success"})

# ส่วนสำหรับรันโปรแกรม (ปรับปรุงเพื่อ Render)
if __name__ == '__main__':
    # ดึงค่า PORT ที่ Render กำหนดให้ ถ้าไม่มี (รันในคอมตัวเอง) ให้ใช้ 5000
    port = int(os.environ.get("PORT", 5000))
    # ตั้ง host='0.0.0.0' เพื่อให้คนภายนอกเข้าดูเว็บได้
    app.run(host='0.0.0.0', port=port)