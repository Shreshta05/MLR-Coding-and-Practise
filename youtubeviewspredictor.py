# 1.simple linear regression - youtube views predictor 

# 1.import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 2.Dataset
data = {
    'ctr': [3.2, 5.8, 2.1, 7.4, 4.5, 6.2, 1.8, 5.1, 3.9, 8.5,
            4.2, 2.8, 6.8, 3.5, 7.1, 2.4, 5.5, 4.8, 6.5, 3.1,
            7.8, 2.6, 5.3, 4.1, 6.9, 3.7, 8.1, 2.2, 5.9, 4.6],

    'total_views': [12000, 28000, 8500, 42000, 19000, 33000, 7000, 24000, 16000, 51000,
                    18000, 11000, 38000, 14500, 44000, 9500, 26000, 21000, 35000, 13000,
                    47000, 10000, 25000, 17500, 40000, 15000, 49000, 8000, 31000, 20000]
}

df = pd.DataFrame(data)

# 3.explore the data
print("First 5 rows of data:")
print(df.head())
print("\nBasic statistics:")
print(df.describe())

# 4.visualize the data
plt.figure(figsize=(10, 6))
plt.scatter(df['ctr'], df['total_views'], c=df['ctr'], cmap='viridis', s = 100, alpha = 0.7, edgecolors='white', linewidth=1.5)
plt.colorbar(label='CTR Intensity')
plt.xlabel('Click Through Rate(CTR)', fontsize = 12, fontweight = 'bold')
plt.ylabel('Total Views', fontsize = 12, fontweight = 'bold')
plt.title('Simple Linear Regression : CTR Vs Total Views', fontsize = 16, fontweight = 'bold')
plt.tight_layout()
plt.legend()
plt.savefig('youtubeviewPredictor1.png', dpi = 150, bbox_inches = 'tight')
plt.show()

# 5.prepare data for training
X = df[['ctr']]
y = df['total_views']

# 6.split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7.create a model and train
model = LinearRegression()
model.fit(X_train, y_train)

# 8. check model parameters
print("\n Model Parameters: ")
print(f" Slope (Coefficient) = {model.coef_[0]:.2f}")
print(f" Intercept =  {model.intercept_:.2f}")

# 9. predictions on test data
y_pred = model.predict(X_train)

# 10.Arjun's question
new_ctr = 8
predicted_views = model.predict([[new_ctr]])
print("\n Arjun's Question : if  CTR is 8%, how many views can he expect? ")
print(f" The Predicted views = {predicted_views[0]:.0f}")

# 11.visualize the regression line
plt.figure(figsize=(10, 6))
plt.scatter(df['ctr'], df['total_views'],
            c='blue', s=100, alpha=0.7, edgecolors='white', linewidth=1.5, label='Actual Data')
X_line = np.linspace(df['ctr'].min(), df['ctr'].max(), 100).reshape(-1, 1)
plt.plot(X_line, model.predict(X_line), color='red', linewidth=2.5, label='Regression Line')
plt.scatter(new_ctr, predicted_views, color = 'green', s = 200, marker = '*', label = 'Our Prediction')
plt.xlabel('Click Through Rate(CTR)', fontsize = 12, fontweight = 'bold')
plt.ylabel('Total Views', fontsize = 12, fontweight = 'bold')
plt.title('Simple Linear Regression : CTR Vs Total Views', fontsize = 16, fontweight = 'bold')
plt.tight_layout()
plt.legend()
plt.savefig('youtubeviewPredictor2.png', dpi = 150, bbox_inches = 'tight')
plt.show()