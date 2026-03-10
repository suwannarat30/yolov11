from pyfirmata2 import Arduino

print("Trying to connect...")
board = Arduino("COM5")   # 🔁 ใส่ COM ที่เช็คได้จริง
print("Connected!")
board.exit()