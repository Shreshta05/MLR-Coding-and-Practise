# 3. MLR - Laptop Price predictor ( 3 features)

# 1.import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#2.Dataset
data = {
    'ram_gb': [4, 8, 4, 16, 8, 8, 4, 16, 8, 16,
               4, 8, 4, 16, 8, 8, 4, 16, 8, 16,
               4, 8, 4, 16, 8, 8, 4, 16, 8, 16],

    'storage_gb': [256, 512, 128, 512, 256, 512, 256, 1024, 256, 512,
                   128, 512, 256, 1024, 256, 512, 128, 512, 256, 1024,
                   256, 512, 128, 512, 256, 512, 256, 1024, 256, 512],

    'processor_ghz': [2.1, 2.8, 1.8, 3.2, 2.4, 3.0, 2.0, 3.5, 2.6, 3.0,
                      1.6, 2.8, 2.2, 3.4, 2.5, 2.9, 1.9, 3.1, 2.3, 3.6,
                      2.0, 2.7, 1.7, 3.3, 2.4, 3.0, 2.1, 3.5, 2.6, 3.2],

    'price_inr': [28000, 45000, 22000, 72000, 38000, 52000, 26000, 95000, 42000, 68000,
                  20000, 48000, 29000, 88000, 40000, 50000, 23000, 70000, 36000, 98000,
                  25000, 46000, 21000, 75000, 39000, 53000, 27000, 92000, 43000, 73000]
}

df = pd.DataFrame(data)

#3. explore the data
print("First 5 rows of data:")
print(df.head())
print("\nBasic statistics:")
print(df.describe())

# 4.visualize the data
fig, axis = plt.subplots(1, 3, figsize = (14, 4))

scatter1 = axis[0].scatter(df['ram_gb'], df['price_inr'], c = df['price_inr'], cmap = "coolwarm", s=100, alpha=0.7, edgecolors='white', linewidth=1.5)
axis[0].set_xlabel('RAM (in GB)', fontsize=12, fontweight='bold')
axis[0].set_ylabel('Price (INR)', fontsize=12, fontweight='bold')
axis[0].set_title('RAM vs Price', fontsize=14, fontweight='bold')
plt.colorbar(scatter1, ax=axis[0], label='Price Intensity')

scatter2 = axis[1].scatter(df['storage_gb'], df['price_inr'], c = df['price_inr'], cmap = "viridis", s=100, alpha=0.7, edgecolors='white', linewidth=1.5)
axis[1].set_xlabel('Storage (in Gb)', fontsize=12, fontweight='bold')
axis[1].set_ylabel('Price (INR)', fontsize=12, fontweight='bold')
axis[1].set_title('Storage vs Price', fontsize=14, fontweight='bold')
plt.colorbar(scatter2, ax=axis[1], label='Price Intensity')

scatter3 = axis[2].scatter(df['processor_ghz'], df['price_inr'], c = df['price_inr'], cmap = "plasma", s=100, alpha=0.7, edgecolors='white', linewidth=1.5)
axis[2].set_xlabel('Processor (in ghz)', fontsize=12, fontweight='bold')
axis[2].set_ylabel('Price (INR)', fontsize=12, fontweight='bold')
axis[2].set_title('Processor vs Price', fontsize=14, fontweight='bold')
plt.colorbar(scatter3, ax=axis[2], label='Price Intensity')

plt.suptitle('MLR : Laptop Price Prediction', fontsize = 16, fontweight = 'bold')
plt.tight_layout()
plt.savefig('laptoppricepred.png', dpi = 10, bbox_inches = 'tight')
plt.show()

# 5 prepare data for training
X = df[['ram_gb', 'storage_gb', 'processor_ghz']]
y = df['price_inr']

# 6 split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7 model creation
model = LinearRegression()
model.fit(X_train, y_train)

# 8 check model parameters
print("\n Model Parameters : ")
print(f"  Coefficients : RAM = {model.coef_[0]:.2f}, Storage = {model.coef_[1]:.2f}, Processor = {model.coef_[2]:.2f}")

#9. which factor affects most
ram = abs(model.coef_[0])
storage = abs(model.coef_[1])
processor = abs(model.coef_[2])

if ram > storage and ram > processor:
    print("RAM affects price the most.")
elif storage > ram and storage > processor:
    print("Storage affects price the most.")
else:
    print("Processor affects price the most.")

# 9 predictions on test data
y_pred = model.predict(X_test)

# 10 model accuracy
score = r2_score(y_test, y_pred)
print (f"\n Model Accuracy (R2_Score) = {score:.2f}")

if score >= 0.90:
    print("The model is highly accurate.")
elif score >= 0.70:
    print("The model is reasonably accurate.")
elif score >= 0.50:
    print("The model has moderate accuracy.")
else:
    print("The model has low accuracy.")

#11 Meera's question
new_device = [[16, 512, 3.2]]
predicted_price = model.predict(new_device)
print("\n Meera's Question : for 16GB RAM, 512GB storage, 3.2 GHz processor, what is a fair price?")
print(f" Predicted  Fair Price is = {predicted_price[0]:.0f} INR")

#12 Bonus Question
meera_laptop = [[8, 512, 2.8]]   # RAM, Storage, Processor
predicted_price = model.predict(meera_laptop)

print(f"\n Predicted Price for  laptop with 8GB RAM, 512GB storage, 2.8 GHz : {predicted_price[0]:.0f} INR")

actual_price = 55000

if predicted_price[0] < actual_price:
    print("The laptop is overpriced.")
else:
    print("The laptop is reasonably priced or underpriced.")
