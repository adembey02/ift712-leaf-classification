# IFT712 Classification Project

This project is a classification task using Python and the scikit-learn library. The goal is to test at least six classification methods on a dataset from Kaggle, following best practices in cross-validation and hyperparameter tuning.

## Project Structure

```
ift712-classification-project
├── data
│   ├── raw                # Raw data files downloaded from Kaggle
│   └── processed          # Processed data files ready for analysis
├── notebooks
│   └── analysis.ipynb     # Jupyter notebook for exploratory data analysis
├── src
│   ├── __init__.py        # Marks the src directory as a Python package
│   ├── main.py            # Main entry point of the application
│   ├── classifiers
│   │   ├── __init__.py    # Marks the classifiers directory as a Python package
│   │   └── classifier_base.py # Base class for classifiers
│   ├── data
│   │   ├── __init__.py    # Marks the data directory as a Python package
│   │   └── loader.py      # Functions to load data from the raw directory
│   ├── preprocessing
│   │   ├── __init__.py    # Marks the preprocessing directory as a Python package
│   │   └── preprocessor.py # Class/functions for data preprocessing
│   ├── evaluation
│   │   ├── __init__.py    # Marks the evaluation directory as a Python package
│   │   └── evaluator.py    # Functions to evaluate classifier performance
│   └── utils
│       ├── __init__.py    # Marks the utils directory as a Python package
│       └── config.py      # Configuration settings for the project
├── tests
│   ├── __init__.py        # Marks the tests directory as a Python package
│   └── test_classifiers.py # Unit tests for classifier implementations
├── .gitignore              # Files and directories to ignore by Git
├── requirements.txt        # Lists project dependencies
├── setup.py                # Packaging and dependency management
└── README.md               # Project documentation
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd ift712-classification-project
pip install -r requirements.txt
```

## Usage

1. Load the data using the `loader.py` module.
2. Preprocess the data using the `preprocessor.py` module.
3. Train classifiers using the `classifier_base.py` module.
4. Evaluate the classifiers using the `evaluator.py` module.
5. Explore the data and results in the `analysis.ipynb` notebook.

## Contributing

Contributions are welcome! Please create a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.