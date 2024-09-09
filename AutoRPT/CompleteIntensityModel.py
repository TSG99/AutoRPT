#!/usr/bin/env python
# coding: utf-8

# In[5]:


import torch
import os
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# Define the neural network architecture
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)  # RNN layer
        self.fc = nn.Linear(hidden_size, 1)  # Linear layer for classification

    def forward(self, x):
        _, hidden = self.rnn(x)
        output = self.fc(hidden.squeeze(0))
        return output

class IBM:
    
    def ibm_model(self, file_name):
        # Load the model
        model = RNNClassifier(input_size=4, hidden_size=32)  # Adjust parameters accordingly
        
        state_start = os.getcwd()
        state_full = os.path.join(state_start, 'model_path', 'IBM.pth')
        model.load_state_dict(torch.load(state_full))
        model.eval()  # Set the model to evaluation mode

        # Load and preprocess new data
        new_data = pd.read_csv(file_name)

        # Extract input features from new data
        X_new = new_data.iloc[:, [3, 4, 5, 6]].values  # Adjust column indices if necessary

        # Handle NaNs in the input data by replacing them with a fixed value (e.g., 0)
        X_new = pd.DataFrame(X_new).fillna(0).values

        # Convert to PyTorch tensors
        X_new_tensor = torch.tensor(X_new, dtype=torch.float32)

        # Create DataLoader for the new data
        new_loader = DataLoader(TensorDataset(X_new_tensor), batch_size=32, shuffle=False)

        # Initialize list to store predictions
        new_predictions = []

        # Set your custom threshold
        custom_threshold = 0.13

        # Run the model on the new data
        for inputs in new_loader:
            inputs = inputs[0].unsqueeze(1)  # Adjust input shape
            outputs = model(inputs)

            # Apply sigmoid to convert logits to probabilities
            probabilities = torch.sigmoid(outputs)

            # Convert probabilities to binary predictions using the custom threshold
            predictions = (probabilities >= custom_threshold).int()

            # Handle NaN values in predictions
            predictions[torch.isnan(probabilities)] = 0  # Example: Set NaN predictions to class 0

            # Store predictions
            new_predictions.extend(predictions.squeeze().detach().numpy())

        # To CSV
        new_data['predictions'] = new_predictions

        out_file_name = os.path.basename(file_name)
        out_file_start = os.getcwd()
        out_file = os.path.join(out_file_start, "csv_outputs", "Predictions_IBM_" + out_file_name)

        new_data.to_csv(out_file, index=False)

        
        
        return out_file