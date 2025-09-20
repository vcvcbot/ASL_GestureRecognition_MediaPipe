# Hand Gesture Recognition Project

## Project Introduction

This project aims to develop a hand gesture recognition system based on MediaPipe and scikit-learn. By extracting key skeletal landmarks from hand images and training a Random Forest Classifier, the system achieves automatic recognition of various hand gestures. This project is suitable for applications requiring gesture-controlled interfaces, assistive communication tools, or interactive applications.

## Features

*   **MediaPipe-based Skeleton Extraction**: Utilizes the MediaPipe library for efficient and accurate detection and extraction of 21 hand skeletal landmarks from hand images.
*   **Random Forest Classifier**: Employs the Random Forest algorithm from scikit-learn for model training, achieving high-accuracy gesture classification.
*   **Dataset Management**: Supports automatic decompression of training datasets from a zip file and subsequent preprocessing.
*   **Model Persistence**: The trained model can be saved for convenient loading and reuse.
*   **Visualizations**: Provides visualizations of the skeleton extraction process to aid in understanding how the model identifies gestures.

## Project Structure

```
.
├── skeleton_model.ipynb        # Core Jupyter Notebook code, including data extraction, model training, and evaluation
├── handgesturefinal.zip        # Compressed file containing the training image dataset
├── F.png                       # Example hand gesture image (e.g., gesture 'F')
└── README.md                   # This README file
```

After decompressing `handgesturefinal.zip`, the following folder structure will be created:

```
.
└── content
    └── train
        ├── A
        │   ├── A1_jpg.rf.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.jpg
        │   └── ...
        ├── B
        │   ├── B1_jpg.rf.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.jpg
        │   └── ...
        ├── C
        │   ├── C1_jpg.rf.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.jpg
        │   └── ...
        └── ... # Other gesture category folders
```

## Setup and Installation

### Prerequisites

*   Python 3.8+
*   pip

### Install Dependencies

Before running the Jupyter Notebook, ensure all necessary Python packages are installed. You can install them using the following command:

```bash
pip install mediapipe scikit-learn opencv-python matplotlib numpy seaborn joblib
```

Alternatively, if you are running in a Colab environment, the Notebook already includes installation commands:

```python
!pip install mediapipe
!pip install scikit-learn
```

## Usage

1.  **Clone the Project**: Clone this repository to your local machine:

    ```bash
    git clone <Your GitHub Project URL>
    cd <Your Project Folder Name>
    ```

2.  **Place the Dataset**: Place the `handgesturefinal.zip` file in the root directory of the project.

3.  **Run Jupyter Notebook**: 

    Launch Jupyter Lab or Jupyter Notebook and open the `skeleton_model.ipynb` file.

    ```bash
jupyter notebook skeleton_model.ipynb
    ```

4.  **Execute the Notebook Cells**: 

    Run the code cells in the Notebook sequentially. The Notebook will automatically perform the following steps:
    *   Install necessary Python packages.
    *   Decompress the `handgesturefinal.zip` dataset into the `/content/train` directory.
    *   Extract hand skeletal landmarks from images using MediaPipe.
    *   Train a Random Forest Classifier.
    *   Evaluate model performance and display results.
    *   Save the trained model (`random_forest_model.joblib`) and class mapping (`class_to_idx.pkl`).

## Model Training and Evaluation

The `skeleton_model.ipynb` file details the entire model training workflow:

1.  **Data Loading and Preprocessing**:
    *   The `HandLandmarkExtractor` class loads the image dataset.
    *   MediaPipe processes each image to extract `(x, y)` coordinates of 21 hand landmarks as features.
    *   Extracted features and corresponding labels are converted into NumPy arrays.

2.  **Data Splitting**:
    *   The dataset is split into training and testing sets (default ratio: 80% training, 20% testing).

3.  **Model Training**:
    *   A `RandomForestClassifier` model is initialized.
    *   The model is trained using the training dataset.

4.  **Model Evaluation**:
    *   The model's accuracy is evaluated on the test set.
    *   A Classification Report and Confusion Matrix are generated for detailed analysis of model performance.

5.  **Model Saving**:
    *   The trained model and class index mapping are saved as `random_forest_model.joblib` and `class_to_idx.pkl` for future loading and use.

## Example Results

The Notebook will display sample images with detected skeletal landmarks, as well as performance metrics (e.g., accuracy, classification report, and confusion matrix) after model training.

![Example Gesture F](F.png)
*Figure: An example image of gesture 'F', showing MediaPipe-detected skeletal landmarks.*

## Future Improvements

*   Explore other machine learning or deep learning models to improve recognition accuracy.
*   Expand the dataset to include more gesture categories and diverse backgrounds.
*   Implement real-time gesture recognition capabilities.
*   Optimize the model for deployment on edge devices.

## Contributing

Contributions to this project are welcome! If you have any suggestions or find bugs, please feel free to submit a Pull Request or open an Issue.

## License

This project is licensed under the MIT License. See the `LICENSE` file (if present) for details.

---

**Author**: Manus AI
**Date**: September 20, 2025
