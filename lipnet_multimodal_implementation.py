import os
import cv2
import dlib
import numpy as np
import tensorflow as tf
# from tensorflow.keras.models import load_model
from keras.api.models import load_model
import librosa
# from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Input, concatenate
from keras.api.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Input, concatenate

# from tensorflow.keras.models import Model
# from keras.src.models import Model
from keras.api.models import Model
# from tensorflow.keras.optimizers import Adam
# from keras.src.optimizers import Adam
from keras.api.optimizers import Adam
from scipy.io import wavfile
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import time

# Configuration
VIDEO_PATH = "D:\Codebase\model_comparison\\test_videos\ABOUT.mp4"
SHAPE_PREDICTOR_PATH = "D:\Codebase\model_comparison\pretrained_models\shape_predictor_68_face_landmarks.dat"
LIPNET_MODEL_PATH = "D:\Codebase\model_comparison\pretrained_models\lipnet_weights.h5"

# Constants for video processing
MOUTH_WIDTH = 100
MOUTH_HEIGHT = 50
SEQUENCE_LENGTH = 100  # Adjust based on your model's expected input
VOCAB = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
CHAR_TO_NUM = {char: i for i, char in enumerate(VOCAB)}
NUM_TO_CHAR = {i: char for i, char in enumerate(VOCAB)}

# Initialize face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

class LipReaderSystem:
    def __init__(self, lipnet_model_path):
        # Load the pre-trained LipNet model
        # self.lipnet_model = load_model(lipnet_model_path, compile=False)
        self.lipnet_model = tf.keras.models.load_model(lipnet_model_path, compile=False)
        self.lipnet_model.compile(optimizer=Adam(learning_rate=0.0001), 
                                  loss={'ctc': lambda y_true, y_pred: y_pred})
        
        # Create multimodal model
        self.multimodal_model = self._build_multimodal_model()
        
    def _build_multimodal_model(self):
        """Build a model that combines LipNet features with audio features"""
        
        # We'll assume the LipNet model has an intermediate layer we can tap into
        # to extract features before the final classification
        visual_features = self.lipnet_model.layers[-2].output
        
        # Audio input and processing branch
        audio_input = Input(shape=(SEQUENCE_LENGTH, 13))  # MFCC features
        x = Conv1D(128, kernel_size=5, activation='relu', padding='same')(audio_input)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = LSTM(128, return_sequences=True)(x)
        x = LSTM(64, return_sequences=True)(x)
        audio_features = Dense(128, activation='relu')(x)
        
        # Combine visual and audio features
        combined = concatenate([visual_features, audio_features])
        
        # Final classification layers
        x = Dense(128, activation='relu')(combined)
        x = Dropout(0.3)(x)
        output = Dense(len(VOCAB) + 1, activation='softmax')(x)
        
        # Create the multimodal model
        multimodal_model = Model(
            inputs=[self.lipnet_model.input, audio_input],
            outputs=output
        )
        
        multimodal_model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss={'ctc': lambda y_true, y_pred: y_pred}
        )
        
        return multimodal_model
    
    def extract_mouth_frames(self, video_path, augment=False):
        """Extract mouth ROI from video frames"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = detector(gray)
            if len(faces) > 0:
                # Get facial landmarks
                landmarks = predictor(gray, faces[0])
                
                # Extract mouth coordinates (landmarks 48-68 are for the mouth)
                mouth_points = []
                for i in range(48, 68):
                    mouth_points.append((landmarks.part(i).x, landmarks.part(i).y))
                
                # Determine bounding box for mouth
                x_min = min(pt[0] for pt in mouth_points)
                y_min = min(pt[1] for pt in mouth_points)
                x_max = max(pt[0] for pt in mouth_points)
                y_max = max(pt[1] for pt in mouth_points)
                
                # Add some margin
                margin = 10
                x_min = max(0, x_min - margin)
                y_min = max(0, y_min - margin)
                x_max = min(frame.shape[1], x_max + margin)
                y_max = min(frame.shape[0], y_max + margin)
                
                # Extract mouth ROI
                mouth_roi = gray[y_min:y_max, x_min:x_max]
                
                # Resize to fixed dimensions
                if mouth_roi.size > 0:
                    mouth_roi = cv2.resize(mouth_roi, (MOUTH_WIDTH, MOUTH_HEIGHT))
                    
                    # Normalize
                    mouth_roi = mouth_roi / 255.0
                    
                    frames.append(mouth_roi)
        
        cap.release()
        
        # If we need to trim or pad to reach SEQUENCE_LENGTH
        if len(frames) < SEQUENCE_LENGTH:
            # Pad with zeros
            padding = [np.zeros((MOUTH_HEIGHT, MOUTH_WIDTH)) for _ in range(SEQUENCE_LENGTH - len(frames))]
            frames.extend(padding)
        elif len(frames) > SEQUENCE_LENGTH:
            # Trim to SEQUENCE_LENGTH
            frames = frames[:SEQUENCE_LENGTH]
            
        return np.array(frames)
    
    def extract_audio_features(self, video_path):
        """Extract MFCC features from video's audio"""
        # Extract audio from video
        temp_audio_path = "temp_audio.wav"
        os.system(f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {temp_audio_path} -y")
        
        # Load audio and extract MFCCs
        y, sr = librosa.load(temp_audio_path, sr=16000)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Transpose to get time steps as first dimension
        mfccs = mfccs.T
        
        # Match the sequence length
        if mfccs.shape[0] < SEQUENCE_LENGTH:
            # Pad with zeros
            padding = np.zeros((SEQUENCE_LENGTH - mfccs.shape[0], mfccs.shape[1]))
            mfccs = np.vstack((mfccs, padding))
        elif mfccs.shape[0] > SEQUENCE_LENGTH:
            # Trim to SEQUENCE_LENGTH
            mfccs = mfccs[:SEQUENCE_LENGTH, :]
            
        # Clean up
        os.remove(temp_audio_path)
        
        return mfccs
    
    def predict_with_lipnet(self, video_path):
        """Predict text using only the visual LipNet model"""
        start_time = time.time()
        
        # Extract mouth frames
        mouth_frames = self.extract_mouth_frames(video_path)
        
        # Add batch dimension
        mouth_frames = np.expand_dims(mouth_frames, axis=0)
        
        # Predict
        prediction = self.lipnet_model.predict(mouth_frames)
        
        # Decode prediction
        decoded_text = self.decode_prediction(prediction[0])
        
        elapsed_time = time.time() - start_time
        
        return {
            'text': decoded_text,
            'confidence': np.max(prediction),
            'processing_time': elapsed_time
        }
    
    def predict_with_multimodal(self, video_path):
        """Predict text using both visual and audio features"""
        start_time = time.time()
        
        # Extract mouth frames
        mouth_frames = self.extract_mouth_frames(video_path)
        mouth_frames = np.expand_dims(mouth_frames, axis=0)
        
        # Extract audio features
        audio_features = self.extract_audio_features(video_path)
        audio_features = np.expand_dims(audio_features, axis=0)
        
        # Predict with multimodal model
        prediction = self.multimodal_model.predict([mouth_frames, audio_features])
        
        # Decode prediction
        decoded_text = self.decode_prediction(prediction[0])
        
        elapsed_time = time.time() - start_time
        
        return {
            'text': decoded_text,
            'confidence': np.max(prediction),
            'processing_time': elapsed_time
        }
    
    def decode_prediction(self, prediction):
        """Convert model output to text"""
        # Get the most likely character at each position
        indices = np.argmax(prediction, axis=1)
        
        # Group repeated characters
        grouped_indices = []
        for i, idx in enumerate(indices):
            if i == 0 or idx != indices[i-1]:
                grouped_indices.append(idx)
        
        # Remove blanks (typically represented by index 0)
        pred_indices = [idx for idx in grouped_indices if idx != 0]
        
        # Convert indices to characters
        predicted_text = ''.join([NUM_TO_CHAR.get(idx, '') for idx in pred_indices])
        
        return predicted_text
    
    def evaluate_models(self, video_path, ground_truth_text):
        """Compare performance of visual-only and multimodal approaches"""
        # Predict with visual-only LipNet
        visual_result = self.predict_with_lipnet(video_path)
        
        # Predict with multimodal approach
        multimodal_result = self.predict_with_multimodal(video_path)
        
        # Calculate metrics
        visual_accuracy = calculate_word_accuracy(visual_result['text'], ground_truth_text)
        multimodal_accuracy = calculate_word_accuracy(multimodal_result['text'], ground_truth_text)
        
        # Calculate character error rate
        visual_cer = calculate_cer(visual_result['text'], ground_truth_text)
        multimodal_cer = calculate_cer(multimodal_result['text'], ground_truth_text)
        
        comparison = {
            'visual_only': {
                'predicted_text': visual_result['text'],
                'confidence': visual_result['confidence'],
                'word_accuracy': visual_accuracy,
                'character_error_rate': visual_cer,
                'processing_time': visual_result['processing_time']
            },
            'multimodal': {
                'predicted_text': multimodal_result['text'],
                'confidence': multimodal_result['confidence'],
                'word_accuracy': multimodal_accuracy,
                'character_error_rate': multimodal_cer,
                'processing_time': multimodal_result['processing_time']
            }
        }
        
        return comparison
    
    def visualize_comparison(self, comparison_results):
        """Visualize the performance comparison between approaches"""
        metrics = ['Word Accuracy', 'Character Error Rate', 'Processing Time', 'Confidence']
        visual_values = [
            comparison_results['visual_only']['word_accuracy'], 
            comparison_results['visual_only']['character_error_rate'],
            comparison_results['visual_only']['processing_time'],
            comparison_results['visual_only']['confidence']
        ]
        multimodal_values = [
            comparison_results['multimodal']['word_accuracy'], 
            comparison_results['multimodal']['character_error_rate'],
            comparison_results['multimodal']['processing_time'],
            comparison_results['multimodal']['confidence']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        visual_bars = ax.bar(x - width/2, visual_values, width, label='Visual Only')
        multimodal_bars = ax.bar(x + width/2, multimodal_values, width, label='Multimodal')
        
        ax.set_ylabel('Value')
        ax.set_title('Performance Comparison: Visual-Only vs Multimodal')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('performance_comparison.png')
        plt.show()
        
        # Also print the predicted texts
        print("Visual-Only Prediction:", comparison_results['visual_only']['predicted_text'])
        print("Multimodal Prediction:", comparison_results['multimodal']['predicted_text'])
        print("Ground Truth:", ground_truth_text)

def calculate_word_accuracy(predicted_text, ground_truth):
    """Calculate word-level accuracy"""
    pred_words = predicted_text.lower().split()
    truth_words = ground_truth.lower().split()
    
    correct = sum(1 for pw, tw in zip(pred_words, truth_words) if pw == tw)
    total = max(len(pred_words), len(truth_words))
    
    return correct / total if total > 0 else 0

def calculate_cer(predicted_text, ground_truth):
    """Calculate character error rate"""
    pred_chars = list(predicted_text.lower())
    truth_chars = list(ground_truth.lower())
    
    # Levenshtein distance
    distances = np.zeros((len(pred_chars) + 1, len(truth_chars) + 1))
    
    # Initialize first row and column
    for i in range(len(pred_chars) + 1):
        distances[i, 0] = i
    for j in range(len(truth_chars) + 1):
        distances[0, j] = j
    
    # Calculate distances
    for i in range(1, len(pred_chars) + 1):
        for j in range(1, len(truth_chars) + 1):
            if pred_chars[i-1] == truth_chars[j-1]:
                distances[i, j] = distances[i-1, j-1]
            else:
                distances[i, j] = min(
                    distances[i-1, j] + 1,    # deletion
                    distances[i, j-1] + 1,    # insertion
                    distances[i-1, j-1] + 1   # substitution
                )
    
    # CER = Levenshtein distance / length of ground truth
    cer = distances[len(pred_chars), len(truth_chars)] / len(truth_chars) if len(truth_chars) > 0 else 0
    
    return cer

# Example usage
if __name__ == "__main__":
    # Initialize the system
    lip_reader = LipReaderSystem(LIPNET_MODEL_PATH)
    
    # Example ground truth (you would replace this with actual ground truth)
    ground_truth_text = "ABOUT"
    
    # Run evaluation
    comparison_results = lip_reader.evaluate_models(VIDEO_PATH, ground_truth_text)
    
    # Print comparison
    print("COMPARISON RESULTS:")
    print("-" * 50)
    print("Visual-Only Approach:")
    print(f"Predicted Text: {comparison_results['visual_only']['predicted_text']}")
    print(f"Word Accuracy: {comparison_results['visual_only']['word_accuracy']:.2f}")
    print(f"Character Error Rate: {comparison_results['visual_only']['character_error_rate']:.2f}")
    print(f"Processing Time: {comparison_results['visual_only']['processing_time']:.2f} seconds")
    print(f"Confidence: {comparison_results['visual_only']['confidence']:.2f}")
    print("-" * 50)
    print("Multimodal Approach:")
    print(f"Predicted Text: {comparison_results['multimodal']['predicted_text']}")
    print(f"Word Accuracy: {comparison_results['multimodal']['word_accuracy']:.2f}")
    print(f"Character Error Rate: {comparison_results['multimodal']['character_error_rate']:.2f}")
    print(f"Processing Time: {comparison_results['multimodal']['processing_time']:.2f} seconds")
    print(f"Confidence: {comparison_results['multimodal']['confidence']:.2f}")
    
    # Visualize the comparison
    lip_reader.visualize_comparison(comparison_results)
