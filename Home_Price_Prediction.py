import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set input
X = [139, 203, 143, 160, 180, 194, 250, 263, 330] #พื้นที่ใช้สอย ตารางเมตร
x1 = [4, 6, 5, 6, 6, 7, 6, 8, 8] #จำนวนห้อง
x2 = [18.7, 54.7, 50.7, 64, 53.9, 63.5, 103.7, 69.6, 72.7] #ขนาดที่ดิน ตารางวา
x3 = [17952, 19653, 32004, 37438, 34944, 38660, 34000, 38289, 39081] #ราคาต่อ ตร.ม.
Y = [2490000, 3990000, 4590000, 5990000, 6290000, 7500000, 8500000, 10070000, 12900000] #ราคาบ้าน

plt.scatter(X, Y)
plt.xlabel("Factor ")  # เพิ่มป้ายแกน x
plt.ylabel("Home price(Baht)")  # เพิ่มป้ายแกน y
plt.title("Home price prediction")
plt.show()

# Normalization
x_max, x_min = max(X), min(X)
x1_max, x1_min = max(x1), min(x1)
x2_max, x2_min = max(x2), min(x2)
x3_max, x3_min = max(x3), min(x3)
y_max, y_min = max(Y), min(Y)

x_normalize = np.array([(xi - x_min) / (x_max - x_min) for xi in X])
x1_normalize = np.array([(x1i - x1_min) / (x1_max - x1_min) for x1i in x1])
x2_normalize = np.array([(x2i - x2_min) / (x2_max - x2_min) for x2i in x2])
x3_normalize = np.array([(x3i - x3_min) / (x3_max - x3_min) for x3i in x3])
y_normalize = np.array([(yi - y_min) / (y_max - y_min) for yi in Y])

plt.scatter(x_normalize, y_normalize)
plt.xlabel("Factor (Normalized)")  # เพิ่มป้ายแกน x
plt.ylabel("Home price (Normalized)")  # เพิ่มป้ายแกน y
plt.title("Home price prediction")
plt.show()

# Build a model
θ0, θ1, θ2, θ3, θ4 = 2.0, 5.0, 2.0, 1.5, 0.5
a = 0.095 # Learning rate
epochs = 600
results = []

for epoch in range(epochs):
    # คำนวณ y_predict
    y_predict = θ0 + (θ1 * x_normalize) + (θ2 * x1_normalize) + (θ3 * x2_normalize) + (θ4 * x3_normalize)

    # คำนวณค่าความคลาดเคลื่อน
    SE = 0.5 * (y_normalize - y_predict) ** 2
    SSE = sum(SE)

    # คำนวณค่าการเปลี่ยนแปลงของ θ
    D_θ0 = sum(-1 * (y_normalize - y_predict))
    D_θ1 = sum(-1 * x_normalize * (y_normalize - y_predict))
    D_θ2 = sum(-1 * x1_normalize * (y_normalize - y_predict))
    D_θ3 = sum(-1 * x2_normalize * (y_normalize - y_predict))
    D_θ4 = sum(-1 * x3_normalize * (y_normalize - y_predict))

    # อัปเดตค่า θ
    θ0 -= a * D_θ0
    θ1 -= a * D_θ1
    θ2 -= a * D_θ2
    θ3 -= a * D_θ3
    θ4 -= a * D_θ4

    # บันทึกค่าที่ใช้แสดงผล
    for i in range(len(x_normalize)):
        results.append([epoch + 1, x_normalize[i], x1_normalize[i], x2_normalize[i], x3_normalize[i],y_normalize[i],y_predict[i], SE[i], θ0, θ1, θ2, θ3, θ4])

    print(f"Epoch {epoch+1}: Error = {SSE:.6f}, θ0 = {D_θ0:.6f}, θ1 = {D_θ1:.6f}, θ2 = {D_θ2:.6f}, θ3 = {D_θ3:.6f}, θ4 = {D_θ4:.6f}")

θ1_denormalize = θ1 * (y_max - y_min) / (x_max - x_min)
θ2_denormalize = θ2 * (y_max - y_min) / (x1_max - x1_min)
θ3_denormalize = θ3 * (y_max - y_min) / (x2_max - x2_min)
θ4_denormalize = θ4 * (y_max - y_min) / (x3_max - x3_min)
θ0_denormalize = (θ0 * (y_max - y_min) + y_min) - (θ1_denormalize * x_min) - (θ2_denormalize * x1_min) - (θ3_denormalize * x2_min) - (θ4_denormalize * x3_min)

# Print final equation
print(f"สมการเส้นตรงที่ได้จาก Linear Regression:")
print(f"y = {θ0_denormalize:.2f} + ({θ1_denormalize:.2f} * x) + ({θ2_denormalize:.2f} * x1) + ({θ3_denormalize:.2f} * x2) + ({θ4_denormalize:.2f} * x3)")

# พล็อตเส้น Linear Regression
plt.scatter(x_normalize, y_normalize, label="Actual Data")
#plt.plot(x, y_predict, color='red', label="Regression Line")
plt.title("Home price prediction")
plt.plot([min(x_normalize), max(x_normalize)], [min(y_predict), max(y_predict)], color='red')
plt.legend()
plt.show()

# Create DataFrame
df = pd.DataFrame(results, columns=["Epoch", "x_normalize", "x1_normalize", "x2_normalize", "x3_normalize","y_normalize", "y_predict", "SE", "θ0", "θ1", "θ2", "θ3", "θ4"])
# แสดงตารางผลลัพธ์
print("\nตารางค่าที่ได้จากการคำนวณ:")
print(df)