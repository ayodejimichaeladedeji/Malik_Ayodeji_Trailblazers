# Email Classification using AI

## Project Overview
This project aims to create an AI-driven email classification system using various machine learning and natural language processing (NLP) techniques. The system processes customer emails, translates them if necessary, removes noise, and classifies them into predefined categories to automate responses and improve customer service efficiency.

## Features
- **Data Loading and Preprocessing:** Loading datasets and cleaning data.
- **Text Translation:** Translating non-English emails into English.
- **Noise Reduction:** Removing irrelevant content from emails.
- **Feature Extraction:** Converting text into numerical features using TF-IDF.
- **Model Training and Evaluation:** Training an XGBoost classifier and evaluating its performance.
- **Performance Visualization:** Visualizing the confusion matrix and performance metrics.

## Requirements
To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
```
## Directory Structure
```
├── .github/
│   └── workflows/
│       └── python-app.yml
├── src/
│   └── email_classifier.py
├── tests/
│   └── test_translate.py
├── .gitignore
├── AppGallery.csv
├── README.md
├── requirements.txt
```
## Description of Files and Directories
- **.github/workflows/python-app.yml:**  GitHub Actions configuration for CI/CD.
- **src/email_classifier.py::** Main script for email classification.
- **tests/test_translate.py:** Unit tests for the translation function.
- **AppGallery.csv:**  Sample dataset.
- **requirements.txt:** List of dependencies.

## Getting Started
Clone the Repository
```bash
git clone https://github.com/maaft75/Malik_Ayodeji_Trailblazers.git
cd Malik_Ayodeji_Trailblazers
```
## Running the Classifier
Ensure you have the required dataset (AppGallery.csv) in the project directory. Run the main script:

```bash
python src/email_classifier.py
```
## Running Tests
To run the unit tests, use:

```bash
python -m unittest discover tests
```
## Project Workflow
The workflow follows the Extreme Programming (XP) Agile methodology with the following steps:

Data Selection: Select and preprocess relevant data.
Data Grouping: Organize emails into logical groups.
Language Handling: Translate non-English emails.
Noise Reduction: Remove redundant and irrelevant content.
Feature Extraction: Convert text data to numerical data using TF-IDF.
Imbalanced Data Handling: Balance the dataset.
Model Training and Testing: Train and evaluate the XGBoost model.
Continuous Integration: Use GitHub Actions for automated testing and deployment.
Evaluation and Results
After training the model, it is evaluated using various metrics:

## Accuracy
- F1 Score
- Precision
- Recall
- Confusion Matrix

## Performance Metrics
- Future Work
- Implement advanced data preprocessing techniques.
- Enhance noise reduction methods.
- Improve text representation models.
- Incorporate customer feedback for continuous improvement.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
