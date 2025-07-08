# Drowsiness & Age Detection System

A computer vision-based application that detects drowsiness (eye status) and estimates age from facial images using deep learning models.

## Features

- **Face Detection**: Automatically detects faces in uploaded images using Haar cascade classifiers
- **Eye Status Detection**: Determines if eyes are open or closed to assess drowsiness
- **Age Estimation**: Predicts age groups (0-12, 13-19, 20-35, 36-50, 51+)
- **Multi-Face Support**: Can analyze multiple faces in a single image
- **User-Friendly GUI**: Simple tkinter-based interface for easy image selection and results display

## Project Structure

```
Drowsiness Detector/
├── DrowsyGUI.py                    # Main GUI application
├── Age_model.ipynb                 # Jupyter notebook for age model training
├── Face_Eye_model.ipynb           # Jupyter notebook for eye status model training
├── age_model.h5                   # Trained age prediction model
├── age_model.weights.h5           # Age model weights
├── face_eye_status_model.h5       # Trained eye status detection model
├── ClosedFace/                    # Training data - closed eye images
├── OpenFace/                      # Training data - open eye images
├── dr/                            # Python virtual environment
├── Car.jpg                        # Sample test image
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Setup Instructions

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd "Drowsiness Detector"
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv drowsiness_env
   
   # On Windows:
   drowsiness_env\Scripts\activate
   
   # On macOS/Linux:
   source drowsiness_env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the GUI Application

1. **Start the application**
   ```bash
   python DrowsyGUI.py
   ```

2. **Using the interface**
   - Click "Select Image" button
   - Choose an image file (jpg, png, etc.)
   - The application will:
     - Display the processed image
     - Show a summary popup with detection results
     - Report number of faces detected
     - Indicate awake/asleep status for each person
     - Estimate age group for each detected face

### Model Training (Advanced Users)

The project includes Jupyter notebooks for training custom models:

1. **Age Model Training**
   ```bash
   jupyter notebook Age_model.ipynb
   ```

2. **Eye Status Model Training**
   ```bash
   jupyter notebook Face_Eye_model.ipynb
   ```

## Technical Details

### Models Used

1. **Face Detection**: OpenCV Haar Cascade Classifier
   - Pre-trained classifier for frontal face detection
   - Fast and reliable for real-time applications

2. **Eye Status Detection**: Custom CNN Model
   - Input: 64x64 RGB face images
   - Output: Binary classification (Open/Closed eyes)
   - Architecture: Convolutional Neural Network

3. **Age Estimation**: Custom CNN Model
   - Input: 64x64 RGB face images
   - Output: Age group classification (5 categories)
   - Categories: 0-12, 13-19, 20-35, 36-50, 51+

### Data Processing Pipeline

1. **Image Loading**: OpenCV imread
2. **Face Detection**: Haar cascade with scale factor 1.1
3. **Preprocessing**: 
   - Resize to 64x64 pixels
   - Normalize pixel values (0-1)
   - Convert BGR to RGB
4. **Prediction**: Neural network inference
5. **Results Display**: Tkinter GUI with summary statistics

## Dependencies

- **TensorFlow 2.9.1**: Deep learning framework
- **OpenCV 4.11.0**: Computer vision library
- **Pillow 10.4.0**: Image processing
- **NumPy 2.0.2**: Numerical computing
- **Tkinter**: GUI framework (included with Python)
- **Matplotlib 3.9.4**: Data visualization
- **Scikit-learn 1.6.1**: Machine learning utilities

## Training Data

The project includes training datasets:
- `ClosedFace/`: Images of faces with closed eyes
- `OpenFace/`: Images of faces with open eyes

## Sample Results

The application provides detailed analysis including:
- Total number of faces detected
- Individual face analysis with:
  - Eye status (Open/Closed)
  - Age group prediction with confidence scores
- Summary statistics (awake vs. asleep count)

## Performance Notes

- **Processing Time**: Depends on image size and number of faces
- **Accuracy**: Models trained on specific datasets, performance may vary
- **Memory Usage**: Approximately 500MB for loaded models
- **Supported Formats**: JPG, PNG, BMP, and other OpenCV-supported formats

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure all .h5 model files are in the project directory
   - Check TensorFlow/Keras compatibility

2. **OpenCV Issues**
   - Verify opencv-python installation
   - Some systems may need opencv-python-headless

3. **GUI Not Displaying**
   - Ensure tkinter is available (included with most Python installations)
   - On Linux, may need: `sudo apt-get install python3-tk`

4. **Memory Errors**
   - Reduce image size before processing
   - Close other applications to free memory

### Error Messages

- "Unable to load image": Check file format and path
- "No faces detected": Try different lighting or image quality
- "Prediction error": Check model files and dependencies

## Future Improvements

- Real-time webcam detection
- Mobile app implementation
- Improved age estimation accuracy
- Additional facial analysis features
- Performance optimization for edge devices

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make improvements
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational and research purposes. Please ensure compliance with applicable laws and regulations when using facial recognition technology.

## Acknowledgments

- OpenCV community for computer vision tools
- TensorFlow team for deep learning framework
- Contributors to the training datasets

## Contact

For questions, issues, or contributions, please open an issue in the repository or contact the project maintainer.

---

**Note**: This application processes images locally and does not store or transmit personal data. Always respect privacy and obtain consent when analyzing images of individuals.
