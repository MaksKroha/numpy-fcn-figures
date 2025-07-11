# NumPy FCN Figures

## Description
The NumPy FCN Figures project provides a Python-based implementation for generating and visualizing figures using fully connected neural networks (FCNs) built with NumPy. This project focuses on creating visualizations to demonstrate the behavior of FCNs on various datasets, emphasizing numerical computation and plotting capabilities.

## Project Overview
This project, hosted at [https://github.com/MaksKroha/numpy-fcn-figures](https://github.com/MaksKroha/numpy-fcn-figures), leverages NumPy to build fully connected neural networks from scratch for educational and visualization purposes. It aims to illustrate how FCNs process data and generate insightful figures, such as decision boundaries or regression curves, for various datasets. The project is ideal for learning about neural network fundamentals and visualizing their performance.

## Features
- Implementation of fully connected neural networks using NumPy
- Visualization of FCN outputs (e.g., decision boundaries, regression lines)
- Support for custom datasets and data preprocessing
- Modular code for easy experimentation with network architectures
- Generation of publication-quality plots using Matplotlib

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/MaksKroha/numpy-fcn-figures.git
   ```
2. Navigate to the project directory:
   ```bash
   cd numpy-fcn-figures
   ```
3. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Ensure all dependencies are installed.
2. Run the main script to train the FCN and generate visualizations:
   ```bash
   python src/main.py
   ```
3. Check the `outputs/` directory for generated figures and results.

## Examples
To generate figures for a sample dataset:
```bash
python src/main.py --dataset data/sample_data.csv --plot decision_boundary
```
This command trains an FCN on `sample_data.csv` and creates a decision boundary plot saved in `outputs/`.

Example output:
```
Training completed in 100 epochs
Accuracy: 92.5%
Plot saved to outputs/decision_boundary.png
```

## Dependencies
- Python 3.8+
- numpy>=1.19.0
- matplotlib>=3.4.0
- pandas>=1.3.0
- scikit-learn>=1.0.0

Install all dependencies using:
```bash
pip install -r requirements.txt
```

## Project Structure
```
numpy-fcn-figures/
├── data/                 # Sample datasets for training and visualization
├── src/                  # Source code
│   ├── main.py           # Main script to run the project
│   ├── fcn.py            # Fully connected neural network implementation
│   ├── preprocess.py     # Data preprocessing functions
│   └── visualize.py      # Visualization functions for generating figures
├── outputs/              # Generated plots and results
├── tests/                # Unit tests
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## Testing
To run the unit tests:
```bash
python -m unittest discover tests
```
Tests verify the FCN implementation, data preprocessing, and visualization functions.

## To-do / Roadmap
- Add support for more complex network architectures (e.g., deeper layers)
- Implement additional visualization types (e.g., loss curves, feature importance)
- Support for real-time interactive plotting
- Extend dataset compatibility for larger, more diverse datasets
- Integrate optimization algorithms (e.g., Adam, RMSprop)

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
