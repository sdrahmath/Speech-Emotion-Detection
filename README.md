# Speech Emotion Detection

## Introduction

This project aims to classify emotions from speech audio using machine learning techniques. Recognizing emotions like calm, happy, sad, angry, etc. from voice data has many real-world applications.

## Aim and Purpose

The goal is to train a model to detect emotions like calm, happy, fearful, disgust from audio clips using the RAVDESS dataset. The model can then be used to classify emotions in new audio samples.

## Dataset

The RAVDESS dataset (Ryerson Audio-Visual Database of Emotional Speech and Song) consists of speech audio recordings by professional actors. It contains various emotions expressed in different intensities, making it suitable for emotion detection tasks.

## Methodology

1. **Data Preprocessing**: The audio samples are preprocessed by extracting relevant audio features like Mel-Frequency Cepstral Coefficients (MFCCs), chroma features, and mel spectrograms using the `librosa` library in Python.

2. **Model Selection**: A Multi-Layer Perceptron (MLP) classifier is chosen for this project due to its simplicity and effectiveness in handling tabular data like extracted audio features.

3. **Model Training**: The extracted audio features are used to train the MLP classifier. The dataset is split into training and testing sets to evaluate the model's performance.

4. **Model Evaluation**: The trained model is evaluated using various metrics like accuracy, precision, recall, and F1-score to assess its performance in emotion detection.

## Technology Used

- Python
- Libraries: `librosa`, `sklearn`, `soundfile`, `pickle`, `streamlit`
- Multi-Layer Perceptron (MLP) Classifier for model training

## Output

### Model Training

The model is trained on the RAVDESS dataset containing audio clips of different emotions. Features like MFCCs, chroma, etc., are extracted from each clip and used to train the MLP model.

The model achieved __85%__ accuracy in detecting 4 emotions - calm, happy, fearful, disgust.

### Streamlit Web App

A web app was built using Streamlit to allow testing the model on new audio samples. Users can upload an audio file, and the app extracts features, runs prediction using the trained model, and displays the detected emotion.

![App Demo Gif](https://github.com/sdrahmath/Speech-Emotion-Detection/blob/main/outputs/Speech%20emotion.gif)
<div style="display: flex;">
<img src="https://github.com/sdrahmath/Speech-Emotion-Detection/blob/main/outputs/Screenshot_1.jpg" alt="Image 1" width="450" height="350">
<img src="https://github.com/sdrahmath/Speech-Emotion-Detection/blob/main/outputs/Screenshot_2.jpg" alt="Image 1" width="450" height="350">
</div>

## Conclusion

The machine learning model performs reasonably well in classifying emotions from short audio clips. The web app provides an easy way to test its performance. More training data and experimenting with deep learning models like CNNs or LSTM could further improve accuracy.

## Acknowledgements

- [RAVDESS Dataset](https://smartlaboratory.org/ravdess/)
- Tutorials and documentation for libraries like `librosa`, `soundfile`, `sklearn`, etc.

### Future Improvements

- Use larger datasets for training to improve model generalization.
- Experiment with advanced models like CNN and LSTM to capture sequential patterns in audio data.
- Add more emotions like sad, angry, surprised, etc., to enhance the emotion classification capabilities.
- Deploy the model to a production environment for real-time emotion detection applications.
