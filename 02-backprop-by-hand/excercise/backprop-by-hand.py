#1. imports
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# 2. dataset generation
X , y = make_moons(
    n_samples=500,
    noise=0.20,
    random_state=42
)

# inspecting our dataset

print(f"X shape : {X.shape}")
print(f"y shape : {y.shape}")
print(f"First 5 input samples : \n{X[:5]}")
print(f"First 5 labels : \n{y[:5]}")

# plotting the dataset 
# plt.figure(figsize=(8,6))
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.title("2D Moons Dataset")

# plt.show()

#  train-test-split of dataset(80% , 20%)

X_train, X_test , y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y
)
print("\nAfter Train-Test Split:")

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# ==========================================
# 4. Parameter Initialization
# ==========================================

# Network architecture 
input_size = 2 
hidden_size = 8
output_size = 1

# reproducible weight initialization 
rng = np.random.default_rng(42)

#initialize weights and biases
W1 = rng.normal(0, 0.01, (input_size, hidden_size))
b1 = np.zeros((1,hidden_size))
W2 = rng.normal(0,0.01, (hidden_size, output_size))
b2 = np.zeros((1,output_size))

print("\nParameter Shapes:")

print("W1:", W1.shape)
print("b1:", b1.shape)

print("W2:", W2.shape)
print("b2:", b2.shape)


# ==========================================
# 5. Forward Pass
# ==========================================

def forward_pass(X,W1,b1,W2,b2):
    #hidden layer
    Z1 = X @ W1 + b1
    A1 = np.maximum(0,Z1)

    #output layer
    Z2 = A1 @ W2 + b2
    A2 = 1 / (1+np.exp(-Z2))

    # cache the intermediate value for back-propgation
    cache = (Z1,A1,Z2,A2)

    return A2,cache


# ==========================================
#  Test Forward Pass
# ==========================================

predictions, cache = forward_pass(
    X_train, W1, b1, W2, b2
)

Z1, A1, Z2, A2 = cache

print("\nForward Pass Results:")

print("Z1 shape:", Z1.shape)
print("A1 shape:", A1.shape)
print("Z2 shape:", Z2.shape)
print("A2 shape:", A2.shape)

print("\nFirst 5 predictions:")
print(predictions[:5])


# ==========================================
# 6. Binary Cross Entropy Loss
# ==========================================

def binary_cross_entropy(y_true, y_pred):
    #avoid log(0)
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1-epsilon)

    loss = -np.mean(
        y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred)
        )

    return loss 



# ==========================================
# Test Loss Function
# ==========================================

loss = binary_cross_entropy(y_train.reshape(-1, 1), predictions)

print("\nInitial Training Loss:", loss)



# ==========================================
# 7. Backward-pass function
# ==========================================

def backward_pass(X, y, W2 , cache):
    Z1 , A1 , Z2 , A2 = cache

    N = X.shape[0]
    Y = y.reshape(-1,1)

    # Output layer delta (BCE + Sigmoid)
    dZ2 = (A2 - Y)/N

    # Output layer gradients
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    #Hidden layer delta
    dA1 = dZ2 @ W2.T
    dZ1 =  dA1*(Z1>0)

    #Hidden layer gradient
    dW1 = X.T @ dZ1 
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2


# ==========================================
# Test Backward-pass function
# ==========================================


dW1, db1, dW2, db2 = backward_pass(
    X_train, y_train, W2, cache
)

print("\nBackward Pass Gradient Shapes:")
print("dW1:", dW1.shape)
print("db1:", db1.shape)
print("dW2:", dW2.shape)
print("db2:", db2.shape)


# ==========================================
# 8. Gradient Descent
# ==========================================

def update_parameters(W1, b1, W2, b2,
                      dW1, db1, dW2, db2,
                      learning_rate):

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    return W1, b1, W2, b2


# ==========================================
# Test Gradient Descent
# ==========================================

learning_rate = 0.1

W1, b1, W2, b2 = update_parameters(
    W1, b1, W2, b2,
    dW1, db1, dW2, db2,
    learning_rate
)

print("\nParameters updated successfully!")
print("W1:", W1.shape)
print("W2:", W2.shape)


# ==========================================
# 9. Numerical Gradient Checking
# ==========================================

def calculate_loss_for_parameters(X,y,W1,b1,W2,b2):

    predictions , _ = forward_pass(X,W1,b1,W2,b2)

    loss = binary_cross_entropy(y.reshape(-1,1), predictions)

    return loss


# ==========================================
# Test Numerical Gradient Checking
# ==========================================

test_loss = calculate_loss_for_parameters(
    X_train, y_train, W1, b1, W2, b2
)

print("Loss for current parameters:", test_loss)




# Check one weight: W2[0, 0]

epsilon = 1e-5

# Save original weight
original_weight = W2[0, 0]

# Calculate loss after increasing weight
W2[0, 0] = original_weight + epsilon
loss_plus = calculate_loss_for_parameters(
    X_train[:10], y_train[:10], W1, b1, W2, b2
)

# Calculate loss after decreasing weight
W2[0, 0] = original_weight - epsilon
loss_minus = calculate_loss_for_parameters(
    X_train[:10], y_train[:10], W1, b1, W2, b2
)

# Restore original weight
W2[0, 0] = original_weight

# Numerical gradient
numerical_gradient = (loss_plus - loss_minus) / (2 * epsilon)

# Analytical gradient (calculate separately on the same 10 samples)
predictions_check, cache_check = forward_pass(
    X_train[:10], W1, b1, W2, b2
)

gradients = backward_pass(
    X_train[:10], y_train[:10], W2, cache_check
)

dW1_check, db1_check, dW2_check, db2_check = gradients

analytical_gradient = dW2_check[0, 0]

print("Numerical gradient:", numerical_gradient)
print("Analytical gradient:", analytical_gradient)



# ==========================================
# 10. Training Loop
# ==========================================


epochs = 5000
learning_rate = 0.1

loss_history = []

for epoch in range(epochs):
    #1. Forward pass 
    predictions, cache = forward_pass(
        X_train, W1,b1,W2,b2
    )

    #2. calcuate loss
    loss = binary_cross_entropy(
        y_train.reshape(-1,1), predictions
    )

    #3.backward pass
    dW1, db1, dW2, db2 = backward_pass(
        X_train, y_train, W2, cache
    )

    #4. update_parameters
    # 4. Update parameters
    W1, b1, W2, b2 = update_parameters(
        W1, b1, W2, b2,
        dW1, db1, dW2, db2,
        learning_rate
    )

    loss_history.append(loss)
    # Print progress
    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")

print("Number of loss values:", len(loss_history))
print("First 5 losses:", loss_history[:5])
print("Last 5 losses:", loss_history[-5:])


    # ==========================================
# 11. Plot Training Loss
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(loss_history)

plt.xlabel("Epoch")
plt.ylabel("Binary Cross Entropy Loss")
plt.title("Training Loss Curve")

plt.grid(True)
plt.show()



# ==========================================
# 12. Model Evaluation
# ==========================================

# Forward pass on unseen test data
test_predictions, _ = forward_pass(
    X_test, W1, b1, W2, b2
)

# Convert probabilities into binary predictions
predicted_classes = (test_predictions >= 0.5).astype(int)

# Calculate accuracy
accuracy = np.mean(
    predicted_classes.flatten() == y_test
)

# Calculate test loss
test_loss = binary_cross_entropy(
    y_test.reshape(-1, 1), test_predictions
)

print("\nModel Evaluation:")
print(f"Test Loss: {test_loss:.6f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# ==========================================
# 13. Decision Boundary Visualization
# ==========================================

# Define plot boundaries
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

# Create a grid of points
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

# Combine grid coordinates into input samples
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Predict probabilities for every grid point
grid_predictions, _ = forward_pass(
    grid_points, W1, b1, W2, b2
)

# Convert probabilities into a 2D grid
Z = grid_predictions.reshape(xx.shape)

# Plot decision regions and boundary
plt.figure(figsize=(9, 6))

plt.contourf(
    xx, yy, Z,
    levels=[0, 0.5, 1],
    cmap="coolwarm",
    alpha=0.3
)

plt.contour(
    xx, yy, Z,
    levels=[0.5],
    colors="black",
    linewidths=2
)

# Plot original dataset
plt.scatter(
    X[:, 0], X[:, 1],
    c=y,
    cmap="coolwarm",
    edgecolors="black"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Neural Network Decision Boundary")
plt.show()