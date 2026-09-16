# Online Food Ordering Prediction using SVM

This project uses Machine Learning to predict online food ordering outcomes using Support Vector Machine (SVM) classification.

## Dataset

The dataset contains customer information such as:

- Age
- Gender
- Marital Status
- Occupation
- Monthly Income
- Educational Qualifications
- Family Size
- Customer Type
- Latitude
- Longitude
- Pin Code
- Feedback

The target variable is `Output`, which contains `Yes` and `No`.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Support Vector Machine (SVM)
- Google Colab

## Machine Learning Workflow

1. Load the dataset
2. Explore the dataset
3. Remove unwanted columns
4. Convert Monthly Income into numerical values
5. Encode categorical features
6. Separate features and target
7. Split data into training and testing sets
8. Apply StandardScaler
9. Train SVM Classifier
10. Make predictions
11. Evaluate the model
12. Visualize the Confusion Matrix
13. Visualize SVM Decision Boundary

## Model

Support Vector Classifier (SVC) was used for classification.

```python
model = SVC(
    kernel="rbf",
    random_state=42
)
