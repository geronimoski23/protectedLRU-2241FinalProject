# ProtectedLRU-2241FinalProject

This project implements and compares several cache eviction strategies, including a lightweight frequency-informed approach called ProtectedLRU. The simulator evaluates performance across real-world and synthetic access patterns using hit rate and other metrics.

## Features
- Supports LRU, LFU, FIFO, MRU, and ProtectedLRU algorithms

- Plots hit rate and cache efficiency across different cache sizes

- Analyzes reuse distance and workload locality

- Includes real-world trace support (e.g., Citi Bike access logs)

## Usage
1. **Clone the repository**
   ```bash
   git clone https://github.com/geronimoski23/protectedLRU-2241FinalProject.git
   cd protectedLRU-2241FinalProject
2. **Run Script**
   ```bash
   python test-cache.py
3. View Results
