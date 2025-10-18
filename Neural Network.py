import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Genereer niet-lineaire data
np.random.seed(42)
x = np.linspace(-2 * np.pi, 2 * np.pi, 200)
y = np.sin(x) + 0.1 * np.random.randn(200)  # niet-lineair + ruis

# 2. Bouw een eenvoudig neuraal netwerk
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# 3. Compileer en train het model
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=300, verbose=0)

# 4. Maak voorspellingen
y_pred = model.predict(x)

# 5. Plot resultaten
plt.figure(figsize=(8, 5))
plt.scatter(x, y, label='Data', color='blue', s=15)
plt.plot(x, y_pred, label='Model voorspelling', color='red', linewidth=2)
plt.title("Neuraal netwerk leert niet-lineaire relatie (sinus)")
plt.legend()
plt.show()

