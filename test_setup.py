# test_setup.py
import torch
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

def test_environment():
    print(" Testing Environment Setup...")
    print("-" * 50)
    
    # Test PyTorch
    print(f" PyTorch version: {torch.__version__}")
    print(f" CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f" GPU: {torch.cuda.get_device_name(0)}")
        print(f" CUDA version: {torch.version.cuda}")
        print(f" GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Test data fetching
    try:
        ticker = yf.Ticker("BTC-USD")
        data = ticker.history(period="5d")
        print(f" Data fetching: Retrieved {len(data)} days of BTC data")
    except Exception as e:
        print(f" Data fetching error: {e}")
    
    # Test basic tensor operations
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = torch.randn(1000, 1000).to(device)
        y = torch.mm(x, x.t())
        print(f" GPU computation: Matrix multiplication successful on {device}")
    except Exception as e:
        print(f" GPU computation error: {e}")
    
    print("-" * 50)
    print(" Environment setup complete!")

if __name__ == "__main__":
    test_environment()