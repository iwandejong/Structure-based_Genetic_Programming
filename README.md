# Genetic Programming for Classification: Structure-Based vs Standard GP

[![C++](https://img.shields.io/badge/C++-20-blue.svg)](https://isocpp.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)]()

## 🧬 Overview

This project implements and compares two Genetic Programming (GP) approaches for binary classification tasks:
- **Standard Genetic Programming**: Traditional GP with tree-based evolution
- **Structure-Based Genetic Programming**: Enhanced GP that considers structural similarity during evolution

The implementation is tested on the Hepatitis dataset, demonstrating the effectiveness of structure-aware genetic operations in improving classification performance.

## 🚀 Key Features

### Core GP Implementation
- **Tree-based Genetic Programming**: Complete implementation of GP with tree structures
- **Multiple Genetic Operators**: Crossover, mutation, and tournament selection
- **Flexible Terminal Set**: Support for both boolean and numerical features
- **Rich Function Set**: Mathematical operators (+, -, *, /, max, min) and functions (tanh, sin, cos, log)
- **Comparison Operators**: Support for logical comparisons (<, >, <=, >=, ==, !=)

### Structure-Based Enhancements
- **Global Similarity Metrics**: Tree-level structural similarity computation
- **Local Similarity Analysis**: Node-level structural comparison
- **Adaptive Thresholds**: Configurable global and local similarity thresholds
- **Cutoff Depth Control**: Depth-based structural analysis limits

### Data Processing
- **Automatic Data Preprocessing**: Outlier detection and removal
- **Feature Normalization**: Min-max normalization for numerical features
- **Type-aware Processing**: Automatic detection of boolean vs numerical features
- **Missing Value Handling**: Robust handling of incomplete data

## 📊 Performance Metrics

The system evaluates performance using:
- **Balanced Accuracy (BACC)**: Primary evaluation metric for imbalanced datasets
- **Execution Time**: Performance comparison between standard and structure-based approaches
- **Population Fitness**: Generation-wise fitness tracking
- **Tree Complexity**: Depth and size analysis of evolved solutions

## 🛠️ Technical Implementation

### Architecture
```
├── GPStruct.h/cpp          # Main GP implementation
├── GPNodeStruct.h/cpp      # Tree node structure
├── main.cpp               # Execution and comparison logic
├── preprocess.ipynb       # Data preprocessing pipeline
└── hepatitis_cleaned.tsv  # Preprocessed dataset
```

### Key Classes
- **`GPStruct`**: Main genetic programming engine
- **`GPNodeStruct`**: Tree node representation
- **`Dataset`**: Data loading and management

### Genetic Operations
- **Tournament Selection**: Size-configurable tournament selection
- **Subtree Crossover**: Standard subtree exchange
- **Point Mutation**: Random node replacement
- **Structure-aware Crossover**: Similarity-based parent selection

## 📈 Results

The implementation demonstrates:
- **Improved Convergence**: Structure-based GP shows better convergence patterns
- **Enhanced Performance**: Better balanced accuracy on classification tasks
- **Efficient Evolution**: More effective exploration of solution space
- **Robust Solutions**: More generalizable evolved programs

## 🚀 Getting Started

### Prerequisites
- C++20 compatible compiler (GCC or Clang)
- Python 3.7+ (for preprocessing)
- pandas, matplotlib, seaborn (for data analysis)

### Installation & Usage

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd genetic-programming-classification
   ```

2. **Preprocess the data** (optional - cleaned data included)
   ```bash
   jupyter notebook preprocess.ipynb
   ```

3. **Compile the project**
   ```bash
   make compile
   # or for Clang
   make alt
   ```

4. **Run the experiments**
   ```bash
   make run
   ```

5. **Check for memory leaks** (macOS)
   ```bash
   make leaks
   ```

### Configuration

Key parameters can be adjusted in `main.cpp`:
```cpp
// Standard GP parameters
int populationSize = 35;
int maxDepth = 7;
int maxGenerations = 80;
std::vector<double> applicationRates = {0.6, 0.25}; // crossover, mutation

// Structure-based GP parameters
int globalThreshold = 6;
int localThreshold = 8;
int cutoffDepth = 4;
```

## 📋 Dataset

The project uses the **Hepatitis dataset** with the following features:
- **19 features**: Age, sex, clinical symptoms, lab values
- **Binary target**: Hepatitis diagnosis (1/2)
- **Preprocessed**: Outliers removed, features normalized
- **Format**: Tab-separated values (TSV)

## 🔬 Research Context

This implementation is part of academic research in evolutionary computation, specifically exploring:
- **Structural Diversity**: How tree structure affects GP performance
- **Similarity Metrics**: Novel approaches to measuring program similarity
- **Evolutionary Dynamics**: Impact of structure-aware selection on convergence

## 📝 Output

The system provides detailed output including:
- **Performance Comparison**: Side-by-side results of both approaches
- **Timing Analysis**: Execution time for each method
- **Fitness Evolution**: Generation-wise fitness progression
- **Best Solutions**: Final evolved programs for both approaches

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Iwan de Jong** - COS 710 Assignment III

---

⭐ **Star this repository if you find it useful!** 