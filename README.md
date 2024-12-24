# MNIST Dataset Analysis using Machine Learning Techniques

This notebook analyzes the MNIST dataset using various machine learning techniques. The MNIST dataset consists of handwritten digit images, and our goal is to classify these images into their respective digit classes (0-9). The steps followed in this notebook are:

1. **Preparing Dataset**: Load the MNIST dataset, flatten the images, and standardize the dataset using MinMaxScaler.
2. **Visualizing Sample Images**: Display a grid of sample images from the training dataset along with their real labels.
3. **Preparing Models**: Initialize and train several machine learning models on the MNIST dataset, including Logistic Regression, Support Vector Machine (SVM), Decision Tree, XGBoost, and Multi-Layer Perceptron (MLP).
4. **Training and Evaluating Models**: Train the models on the training data and evaluate them on the test data to calculate various performance metrics such as accuracy, F1 score, recall, and precision.
5. **Visualizing Predictions**: Test sample images and show their predicted labels along with the images.
6. **Saving Models**: Save the trained models and the scaler to disk using the `joblib` library.
7. **Loading and Using Saved Models**: Load the saved models from disk and use them to predict the label of a real sample image.

## Requirements

- Python 3.x
- Jupyter Notebook
- TensorFlow
- scikit-learn
- XGBoost
- joblib
- matplotlib
- Pillow
- numpy

## Usage

1. Clone the repository or download the notebook.
2. Install the required libraries.
3. Open the notebook in Jupyter Notebook.
4. Run the cells sequentially to perform the analysis.

## Results

The notebook provides a detailed analysis of the MNIST dataset using various machine learning models. The performance metrics for each model are calculated and displayed. The trained models and scaler are saved to disk for future use.

## License

This project is licensed under the GPL 3.0 License.
