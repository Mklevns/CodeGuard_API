"""
Test script to verify VSCode timeout fixes and false positive filtering optimization.
"""

import requests
import time
import json

def test_timeout_handling():
    """Test the timeout handling improvements for VSCode extension."""
    
    base_url = "https://codeguard.replit.app"
    
    # Large code sample that might trigger timeout
    large_code_sample = '''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import os
import sys
import warnings
warnings.filterwarnings('ignore')

class LargeNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(LargeNeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        
        # Add input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        
        # Add hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        
        # Add output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.relu(layer(x))
            if i % 2 == 0:  # Apply dropout every other layer
                x = self.dropout(x)
        x = self.layers[-1](x)  # Output layer (no activation)
        return x

def load_data_unsafe():
    # This should be flagged as security issue
    with open("data.pkl", "rb") as f:
        data = pickle.load(f)
    return data

def preprocess_data(data):
    # Missing seed - should be caught
    # torch.manual_seed(42)  # Commented out to trigger issue
    
    # Hardcoded path - should be flagged
    output_path = "/tmp/processed_data.csv"
    
    # Using eval - security issue
    config_str = "{'learning_rate': 0.001, 'batch_size': 32}"
    config = eval(config_str)
    
    return data, config

def train_model_with_issues():
    # Multiple issues to test false positive filtering
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Missing random seed for reproducibility
    # np.random.seed(42)  # Commented out
    
    # Hardcoded hyperparameters
    learning_rate = 0.001
    batch_size = 32
    epochs = 100
    
    # Create model
    model = LargeNeuralNetwork([784, 512, 256, 128], 10)
    model.to(device)
    
    # Missing optimizer setup
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop with potential issues
    for epoch in range(epochs):
        # Missing model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Missing optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            # Missing optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')

def evaluate_model():
    # Missing model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            predicted = torch.argmax(output, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = correct / total
    return accuracy

# Missing main guard
if __name__ == "__main__":
    train_model_with_issues()
'''
    
    test_request = {
        "files": [
            {
                "filename": "large_model.py",
                "content": large_code_sample
            }
        ],
        "options": {
            "level": "strict",
            "framework": "pytorch",
            "target": "gpu"
        }
    }
    
    print("Testing Timeout Handling and False Positive Filtering")
    print("=" * 60)
    
    # Test 1: With false positive filtering (should handle timeout gracefully)
    print("\n1. Testing WITH false positive filtering (timeout handling):")
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{base_url}/audit",
            json=test_request,
            headers={"Content-Type": "application/json"},
            timeout=50  # Slightly longer than backend timeout
        )
        
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ SUCCESS in {elapsed_time:.1f}s")
            print(f"   Summary: {data['summary']}")
            print(f"   Issues found: {len(data['issues'])}")
            
            # Check if fallback was used
            if "AI filtering timed out" in data['summary'] or "fallback" in data['summary']:
                print(f"   ⚠ AI filtering timed out, fallback used")
            else:
                print(f"   ✓ AI filtering completed successfully")
                
        else:
            print(f"   ✗ FAILED with status {response.status_code}")
            print(f"   Error: {response.text}")
            
    except requests.exceptions.Timeout:
        elapsed_time = time.time() - start_time
        print(f"   ✗ TIMEOUT after {elapsed_time:.1f}s")
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"   ✗ ERROR after {elapsed_time:.1f}s: {str(e)}")
    
    # Test 2: Without false positive filtering (baseline)
    print("\n2. Testing WITHOUT false positive filtering (baseline):")
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{base_url}/audit/no-filter",
            json=test_request,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ SUCCESS in {elapsed_time:.1f}s")
            print(f"   Summary: {data['summary']}")
            print(f"   Issues found: {len(data['issues'])}")
        else:
            print(f"   ✗ FAILED with status {response.status_code}")
            
    except requests.exceptions.Timeout:
        elapsed_time = time.time() - start_time
        print(f"   ✗ TIMEOUT after {elapsed_time:.1f}s")
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"   ✗ ERROR after {elapsed_time:.1f}s: {str(e)}")
    
    # Test 3: Simple code (should be fast)
    print("\n3. Testing simple code (should be fast):")
    simple_request = {
        "files": [
            {
                "filename": "simple.py",
                "content": "import torch\n\ndef simple_function():\n    return 42"
            }
        ],
        "options": {"level": "standard", "framework": "pytorch"}
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{base_url}/audit",
            json=simple_request,
            headers={"Content-Type": "application/json"},
            timeout=15
        )
        
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ SUCCESS in {elapsed_time:.1f}s")
            print(f"   Issues found: {len(data['issues'])}")
        else:
            print(f"   ✗ FAILED with status {response.status_code}")
            
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"   ✗ ERROR after {elapsed_time:.1f}s: {str(e)}")
    
    print("\n" + "=" * 60)
    print("Timeout testing completed!")

if __name__ == "__main__":
    test_timeout_handling()