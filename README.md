# Predictive Maintenance - Machine Failure Prediction using Machine Learning
  ### Video Demo:  <https://youtu.be/A7WE1fa9Ddc>
  ## Description:
This project investigates the use of Machine Learning to predict equipment failure types based on a dataset obtained from Kaggle <https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification/data?select=predictive_maintenance.csv>. A Decision Tree model is employed to classify the different failure modes.

#### Project Dependencies
* Pandas: Used for data manipulation and analysis.

* scikit-learn: Provides the Decision Tree implementation and other machine learning utilities.

#### Data Preprocessing
The project leverages Pandas to load and pre-process the CSV dataset. This includes:

* Handling missing data using appropriate techniques (e.g., imputation, deletion).

* Reformatting data to desired data types for optimal model training (e.g., string to float).

* Extracting equipment numbers from the "[Product ID]" column.

* Removing outliers such as "Random Failure" for improved model performance.

#### Model Training and Evaluation
The pre-processed data is then split into training and testing sets using train_test_split (default test size: 0.2). A Decision Tree model is trained on the training data. The model's performance is evaluated using metrics such as accuracy and F1 score, achieving a high of 99.6% accuracy and 99.7% F1 score.

#### Visualization
The classification results are visualized using a Confusion Matrix to provide insights into the model's performance and potential areas for improvement.

## Target Users:

This project is intended for data scientists, machine learning engineers, or anyone who needs to preprocess data and train models for analysis or prediction tasks.

#### Installation:
Install the required libraries using pip install <requirements.txt>.

#### Usage:
Import the necessary functions from your project module (e.g., from project import data_preprocessing, from project import train_model).
Use the functions in your data preparation and modeling workflow. Refer to the docstrings within the code for detailed function usage.

## Constraints:

* Limited Model Complexity: Decision trees are relatively simple models that struggle with complex relationships between features. They might not capture non-linear patterns or interactions between features effectively.

* Overfitting: Decision trees are prone to overfitting, especially with high-dimensional data. They can easily memorize the training data without generalizing well to unseen data.

* Interpretability: While decision trees are inherently interpretable due to their tree structure, complex trees can be difficult to understand and analyze.

* Feature Importance: Decision trees provide rudimentary feature importance measures, but they might not accurately reflect the true relationships between features and the target variable.

### Considerations:

* Suitable for Specific Problems: Decision trees can be effective for simple classification problems where the relationships between features are linear or easily captured by a tree structure.

* Fast Training and Prediction: Decision trees are relatively fast to train and predict, making them suitable for real-time applications.

* Exploratory Tool: You can use decision trees as a starting point to explore the data and identify relevant features. The insights gained can be used to guide feature engineering efforts.

### Alternative Approaches for Users:

* Exploring Other Models: Users can explore other classification models like Random Forests (ensemble of decision trees), Support Vector Machines (SVMs), K-Nearest Neighbors (KNN), or Neural Networks (depending on data complexity). These models might achieve better performance for specific problems.

* Feature Engineering: Feature engineering plays a crucial role in improving the performance of any machine learning model. By creating new features or transforming existing ones, users can highlight the underlying structure in the data and make it easier for the model to learn.
