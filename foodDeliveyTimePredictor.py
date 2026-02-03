# 2.Multiple linear regression - food delivery time predictor ( 2 features )

# 1.import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 2.Dataset
data = {
    'distance_km': [2.5, 6.0, 1.2, 8.5, 3.8, 5.2, 1.8, 7.0, 4.5, 9.2,
                    2.0, 6.5, 3.2, 7.8, 4.0, 5.8, 1.5, 8.0, 3.5, 6.8,
                    2.2, 5.5, 4.2, 9.0, 2.8, 7.2, 3.0, 6.2, 4.8, 8.2],

    'prep_time_min': [10, 20, 8, 25, 12, 18, 7, 22, 15, 28,
                      9, 19, 11, 24, 14, 17, 6, 26, 13, 21,
                      10, 16, 14, 27, 11, 23, 12, 18, 15, 25],

    'delivery_time_min': [18, 38, 12, 52, 24, 34, 14, 45, 29, 58,
                          15, 40, 21, 50, 27, 35, 11, 54, 23, 43,
                          17, 32, 26, 56, 19, 47, 20, 37, 30, 53]
}

df = pd.DataFrame(data)

# 3. explore the data
print("First 5 rows of data:")
print(df.head())
print("\nBasic statistics:")
print(df.describe())

# 4.visualize the data
fig, axis = plt.subplots(1, 2, figsize = (14, 4))

scatter1 = axis[0].scatter(df['distance_km'], df['delivery_time_min'], c = df['delivery_time_min'], cmap = "coolwarm", s=100, alpha=0.7, edgecolors='white', linewidth=1.5)
axis[0].set_xlabel('Distance (in kms)', fontsize=12, fontweight='bold')
axis[0].set_ylabel('Delivery Time (in mns)', fontsize=12, fontweight='bold')
axis[0].set_title('Distance Vs Delivery Time', fontsize=14, fontweight='bold')
plt.colorbar(scatter1, ax=axis[0], label='Time Intensity')

scatter2 = axis[1].scatter(df['prep_time_min'], df['delivery_time_min'], c = df['delivery_time_min'], cmap = "viridis", s=100, alpha=0.7, edgecolors='white', linewidth=1.5)
axis[1].set_xlabel('Preparation Time (in mns)', fontsize=12, fontweight='bold')
axis[1].set_ylabel('Delivery Time (in mns)', fontsize=12, fontweight='bold')
axis[1].set_title('Preparation Time Vs Delivery Time ', fontsize=14, fontweight='bold')
plt.colorbar(scatter2, ax=axis[1], label='Time Intensity')

plt.suptitle('Multiple Linear Regression : Delivery Time Estimation', fontsize = 16, fontweight = 'bold')
plt.tight_layout()
plt.savefig('deliverytimepred.png', dpi = 10, bbox_inches = 'tight')
plt.show()

# 5.prepare data for training
X = df[['distance_km', 'prep_time_min']]
y = df['delivery_time_min']

# 6. split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#7. model creation
model = LinearRegression()
model.fit(X_train, y_train)

#8.model coefficients
print("\n Model Parameters : ")
print(f"  Coefficients : Distance = {model.coef_[0]:.2f}, Prep Time = {model.coef_[1]:.2f}")

# 9.which factor affects delivery time more
if abs(model.coef_[0]) > abs(model.coef_[1]):
    print("Distance affects delivery time more.")
else:
    print("Preparation Time affects delivery time more.")


#10.Vikram's question
new_delivery = [[7, 15]]
predicted_time = model.predict(new_delivery)
print("\n Vikram's question : For 7km distance and 15 min prep time , what is the expected delivery time ? ")
print(f" Predicted Time is = {predicted_time[0]:.0f} mns")
