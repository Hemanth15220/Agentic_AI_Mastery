# Day 2: Classical ML vs. Deep Learning (Scikit-Learn to PyTorch)

**Date:** May 13, 2026
**Focus:** Translating a traditional Machine Learning pipeline into a Deep Learning Neural Network using PyTorch.

## 🧠 Core Concept: Trees vs. Networks
When predicting outcomes based on structured tabular data, we have two primary paradigms:

1.  **Classical ML (Random Forest via Scikit-Learn):** * **How it works:** It builds an ensemble of decision trees. It asks a series of True/False questions (e.g., "Is `severity_score` > 5?") to split the data into buckets.
    * **Pros:** Trains in seconds on a CPU. Excellent for tabular data. Highly interpretable.
    * **Cons:** Cannot easily process unstructured data like images, audio, or raw text.
2.  **Deep Learning (Neural Networks via PyTorch):**
    * **How it works:** It uses layers of interconnected "neurons" that pass matrix multiplications through activation functions. It learns by calculating the error (loss) and updating the weights backwards through the network (Backpropagation).
    * **Pros:** The foundational architecture for LLMs, Computer Vision, and Audio processing. Discovers complex, non-linear relationships.
    * **Cons:** Requires massive data, GPUs for training, and acts as a "black box."

## 💻 The Code: A Head-to-Head Comparison
To demonstrate the architectural shift, I built a pipeline to classify a synthetic dataset (e.g., predicting if a case requires immediate clinical intervention based on 3 numerical features). 

First, the Scikit-Learn approach (4 lines of code). Second, the PyTorch approach (defining the network architecture, the loss function, and the training loop).

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# 1. Create Synthetic Tabular Data (e.g., 3 clinical risk features)
# X = [severity_score, past_reports, days_since_last_visit]
np.random.seed(42)
X = np.random.rand(1000, 3) 
# Y = Binary target (1 = High Risk, 0 = Low Risk)
y = (X[:, 0] + X[:, 1] > 1.0).astype(int) 

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# PART A: Classical ML (Scikit-Learn)
# ==========================================
print("--- Training Random Forest ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train) # Training happens entirely in this one line
rf_predictions = rf_model.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_predictions):.4f}\n")

# ==========================================
# PART B: Deep Learning (PyTorch)
# ==========================================
print("--- Training PyTorch Neural Network ---")

# Convert numpy arrays to PyTorch Tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1) # Reshape for PyTorch
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

# Define the Neural Network Architecture
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        # Input layer (3 features) -> Hidden Layer (8 neurons)
        self.layer1 = nn.Linear(3, 8) 
        # Hidden Layer (8 neurons) -> Output Layer (1 neuron)
        self.layer2 = nn.Linear(8, 1) 
        # Activation function to squash output between 0 and 1
        self.sigmoid = nn.Sigmoid()   

    def forward(self, x):
        x = torch.relu(self.layer1(x)) # ReLU activation for hidden layer
        x = self.sigmoid(self.layer2(x))
        return x

# Initialize Model, Loss Function, and Optimizer
pt_model = BinaryClassifier()
criterion = nn.BCELoss() # Binary Cross Entropy Loss
optimizer = optim.Adam(pt_model.parameters(), lr=0.01)

# The Training Loop (Unlike Scikit-Learn, we must manually write the loop)
epochs = 100
for epoch in range(epochs):
    # 1. Forward Pass (Make predictions)
    outputs = pt_model(X_train_tensor)
    
    # 2. Calculate the Error (Loss)
    loss = criterion(outputs, y_train_tensor)
    
    # 3. Backward Pass (Calculate gradients)
    optimizer.zero_grad() # Clear old gradients
    loss.backward()       # Compute new gradients
    
    # 4. Update Weights
    optimizer.step()

# Evaluate the PyTorch Model
with torch.no_grad(): # Turn off gradient tracking for evaluation
    pt_predictions = pt_model(X_test_tensor)
    pt_predictions_classes = pt_predictions.round() # Round probabilities to 0 or 1
    pt_accuracy = accuracy_score(y_test, pt_predictions_classes.numpy())
    print(f"PyTorch NN Accuracy: {pt_accuracy:.4f}")
