import os
import cv2
import dlib
import numpy as np
import tensorflow as tf
from keras.api.models import load_model, Model
from keras.api.layers import (
    Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, LSTM, Dense, Dropout, Input, 
    Bidirectional, GRU, BatchNormalization, Activation, TimeDistributed, Flatten,
    Reshape, MultiHeadAttention, LayerNormalization, Add, Permute, Multiply, 
    Concatenate, Lambda
)
from keras.api.optimizers import Adam
import librosa
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import time
import glob
from scipy import ndimage
import random
import re

# Configuration
GRID_DATASET_PATH = "D:\Codebase\model_comparison\\test_videos\s1"
SHAPE_PREDICTOR_PATH = "D:\Codebase\model_comparison\pretrained_models\shape_predictor_68_face_landmarks.dat"
LIPNET_MODEL_PATH = "D:\Codebase\model_comparison\pretrained_models\lipnet_weights.h5"

# Constants for GRID dataset processing
MOUTH_WIDTH = 100
MOUTH_HEIGHT = 50
SEQUENCE_LENGTH = 75  # GRID dataset videos are 75 frames (3 seconds at 25fps)
VOCAB = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
CHAR_TO_NUM = {char: i for i, char in enumerate(VOCAB)}
NUM_TO_CHAR = {i: char for i, char in enumerate(VOCAB)}

# GRID dataset sentence structure
# Each sentence follows: command + color + preposition + letter + digit + adverb
# For example: "bin blue at f zero again"
GRID_COMMANDS = ['bin', 'lay', 'place', 'set']
GRID_COLORS = ['blue', 'green', 'red', 'white']
GRID_PREPOSITIONS = ['at', 'by', 'in', 'with']
GRID_LETTERS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
GRID_DIGITS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
GRID_ADVERBS = ['again', 'now', 'please', 'soon']

# Initialize face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

class LipReaderSystem:
    def __init__(self, lipnet_model_path):
        self.model_path = lipnet_model_path
        self.lipnet_model = None
        self.char_to_idx = None
        self.idx_to_char = None
        self.load_model()
        # Create multimodal model with cross-modal attention
        self.multimodal_model = self._build_cross_modal_attention_model()

    def load_model(self):
        """Load pre-trained LipNet model or create a new one compatible with GRID dataset"""
        # Define character mapping for GRID dataset
        chars = VOCAB
        self.char_to_idx = CHAR_TO_NUM
        self.idx_to_char = NUM_TO_CHAR

        try:
            self.lipnet_model = tf.keras.models.load_model(self.model_path)
            print("LipNet model loaded successfully")
        except Exception as e:
            print(f"Error loading LipNet model: {e}")
            # Create LipNet model specifically for GRID dataset
            input_shape = (SEQUENCE_LENGTH, 50, 100, 3)  # GRID: 75 frames, 50x100 mouth ROI, RGB
            self.lipnet_model = self._create_lipnet_model(input_shape, len(chars))
            print("Created new LipNet model for GRID dataset")
            
    def _create_lipnet_model(self, input_shape, num_classes):
        """Create LipNet architecture optimized for GRID dataset"""
        inputs = Input(shape=input_shape)
        
        # Spatio-temporal convolutional frontend
        x = TimeDistributed(Conv2D(32, (3, 3), padding='same', activation='relu'))(inputs)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(MaxPooling2D((2, 2)))(x)
        
        x = TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu'))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(MaxPooling2D((2, 2)))(x)
        
        x = TimeDistributed(Conv2D(96, (3, 3), padding='same', activation='relu'))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(MaxPooling2D((2, 2)))(x)
        
        # Flatten spatial dimensions
        x = TimeDistributed(Flatten())(x)
        
        # Bidirectional LSTM
        x = Bidirectional(GRU(256, return_sequences=True))(x)
        x = Dropout(0.2)(x)
        
        x = Bidirectional(GRU(256, return_sequences=True))(x)
        x = Dropout(0.2)(x)
        
        # Output CTC layer
        outputs = Dense(num_classes + 1, activation='softmax')(x)  # +1 for CTC blank
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.0001), loss=self._ctc_loss_placeholder)
        
        return model
    
    def _ctc_loss_placeholder(self, y_true, y_pred):
        """Placeholder for CTC loss"""
        return tf.reduce_mean(y_pred - y_true)


    def _build_cross_modal_attention_model(self):
        """Build a model that uses cross-modal attention between visual and audio features"""

        # Visual branch - reuse architecture from lipnet model
        visual_input = Input(shape=(SEQUENCE_LENGTH, 50, 100, 3))  # GRID mouth ROI dimensions

        # Visual processing
        v = TimeDistributed(Conv2D(32, (3, 3), padding='same', activation='relu'))(visual_input)
        v = TimeDistributed(BatchNormalization())(v)
        v = TimeDistributed(MaxPooling2D((2, 2)))(v)

        v = TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu'))(v)
        v = TimeDistributed(BatchNormalization())(v)
        v = TimeDistributed(MaxPooling2D((2, 2)))(v)

        v = TimeDistributed(Conv2D(96, (3, 3), padding='same', activation='relu'))(v)
        v = TimeDistributed(BatchNormalization())(v)
        v = TimeDistributed(MaxPooling2D((2, 2)))(v)

        v = TimeDistributed(Flatten())(v)

        # Audio branch
        audio_input = Input(shape=(SEQUENCE_LENGTH, 13))  # MFCC features

        # Audio processing - ensure it maintains the same sequence length as visual
        # Removing the pooling layers to maintain sequence length
        a = Conv1D(128, kernel_size=5, activation='relu', padding='same')(audio_input)
        a = BatchNormalization()(a)
        # Removed MaxPooling1D

        a = Conv1D(256, kernel_size=3, activation='relu', padding='same')(a)
        a = BatchNormalization()(a)
        # Removed MaxPooling1D

        a = Bidirectional(LSTM(128, return_sequences=True))(a)

        # Process audio features to match visual feature dimension
        a_proj = Dense(256, activation='relu')(a)
        a_proj = Dropout(0.2)(a_proj)

        # Process visual features to prepare for attention
        v_proj = Dense(256, activation='relu')(v)
        v_proj = Dropout(0.2)(v_proj)

        # 1. Visual-guided attention on audio
        # Attention from visual to audio
        v2a_attention = MultiHeadAttention(num_heads=8, key_dim=32)(
            query=v_proj,
            key=a_proj,
            value=a_proj
        )
        v2a_attention = Dropout(0.1)(v2a_attention)
        v2a_add = Add()([v_proj, v2a_attention])
        v2a_norm = LayerNormalization()(v2a_add)

        # 2. Audio-guided attention on visual
        # Attention from audio to visual
        a2v_attention = MultiHeadAttention(num_heads=8, key_dim=32)(
            query=a_proj,
            key=v_proj,
            value=v_proj
        )
        a2v_attention = Dropout(0.1)(a2v_attention)
        a2v_add = Add()([a_proj, a2v_attention])
        a2v_norm = LayerNormalization()(a2v_add)

        # Make sure both tensors have the same sequence length
        # Option 1: Use time-based average pooling to make sequences match
        if v2a_norm.shape[1] != a2v_norm.shape[1]:
            # If audio sequence is shorter, upsample it
            if a2v_norm.shape[1] < v2a_norm.shape[1]:
                # Use 1D upsampling to match sequence length
                required_size = v2a_norm.shape[1]
                current_size = a2v_norm.shape[1]
                # Reshape and use 1D interpolation
                a2v_norm = Lambda(lambda x: tf.image.resize(
                    tf.expand_dims(x, axis=2),
                    [required_size, 1],
                    method='nearest'
                ))(a2v_norm)
                a2v_norm = Lambda(lambda x: tf.squeeze(x, axis=2))(a2v_norm)
            else:
                # If visual sequence is shorter, upsample it
                required_size = a2v_norm.shape[1]
                current_size = v2a_norm.shape[1]
                # Reshape and use 1D interpolation
                v2a_norm = Lambda(lambda x: tf.image.resize(
                    tf.expand_dims(x, axis=2),
                    [required_size, 1],
                    method='nearest'
                ))(v2a_norm)
                v2a_norm = Lambda(lambda x: tf.squeeze(x, axis=2))(v2a_norm)

        # Combine both attention outputs (cross-modal fusion)
        cross_modal_features = Concatenate()([v2a_norm, a2v_norm])

        # Final processing
        x = Dense(512, activation='relu')(cross_modal_features)
        x = Dropout(0.3)(x)
        x = Bidirectional(GRU(256, return_sequences=True))(x)
        x = Dropout(0.3)(x)

        # Output layer
        output = Dense(len(VOCAB) + 1, activation='softmax')(x)  # +1 for CTC blank

        # Create the multimodal model with cross-modal attention
        multimodal_model = Model(
            inputs=[visual_input, audio_input],
            outputs=output
        )

        # Compile the model
        multimodal_model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy'  # Replace with CTC loss in production
        )

        return multimodal_model

    def extract_mouth_frames(self, video_path, augment=False):
        """Extract and preprocess mouth ROI from video frames for GRID dataset"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
                
                # Add margin around mouth
                margin_x = int((x_max - x_min) * 0.2)
                margin_y = int((y_max - y_min) * 0.2)
                x_min = max(0, x_min - margin_x)
                y_min = max(0, y_min - margin_y)
                x_max = min(frame.shape[1], x_max + margin_x)
                y_max = min(frame.shape[0], y_max + margin_y)
                
                # Extract mouth ROI
                mouth_roi = frame_rgb[y_min:y_max, x_min:x_max]
                
                # Data augmentation if requested
                if augment and random.random() < 0.5:
                    # Random rotation
                    angle = random.uniform(-10, 10)
                    mouth_roi = ndimage.rotate(mouth_roi, angle, reshape=False)
                    
                    # Random horizontal flip
                    if random.random() < 0.5:
                        mouth_roi = cv2.flip(mouth_roi, 1)
                
                # Resize to model's expected dimensions (50x100 for GRID)
                if mouth_roi.size > 0:
                    mouth_roi = cv2.resize(mouth_roi, (MOUTH_WIDTH, MOUTH_HEIGHT))
                    
                    # Normalize pixel values to [0,1]
                    mouth_roi = mouth_roi / 255.0
                    
                    frames.append(mouth_roi)
            else:
                # If no face is detected, add a blank frame (mean face)
                blank_frame = np.zeros((MOUTH_HEIGHT, MOUTH_WIDTH, 3))
                frames.append(blank_frame)
        
        cap.release()
        
        # Convert to numpy array
        frames = np.array(frames)
        
        # Ensure we have exactly SEQUENCE_LENGTH frames
        if len(frames) < SEQUENCE_LENGTH:
            # Pad with blank frames
            padding = np.zeros((SEQUENCE_LENGTH - len(frames), MOUTH_HEIGHT, MOUTH_WIDTH, 3))
            frames = np.vstack([frames, padding])
        elif len(frames) > SEQUENCE_LENGTH:
            # Trim to SEQUENCE_LENGTH
            frames = frames[:SEQUENCE_LENGTH]
            
        return frames
        
    def extract_audio_features(self, video_path):
        """Extract MFCC features from video's audio track optimized for GRID dataset"""
        try:
            # Extract audio with librosa (GRID videos are 3 seconds long)
            y, sr = librosa.load(video_path, sr=16000, mono=True, duration=3.0)
            
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=int(sr/25))  # Match video framerate (25fps)
            mfccs = mfccs.T  # Transpose to get time steps as first dimension
            
            # Apply deltas (first & second derivatives) to capture dynamics
            delta_mfccs = librosa.feature.delta(mfccs.T).T
            delta2_mfccs = librosa.feature.delta(mfccs.T, order=2).T
            
            # Normalize features
            mfccs = (mfccs - np.mean(mfccs, axis=0)) / (np.std(mfccs, axis=0) + 1e-5)
            delta_mfccs = (delta_mfccs - np.mean(delta_mfccs, axis=0)) / (np.std(delta_mfccs, axis=0) + 1e-5)
            delta2_mfccs = (delta2_mfccs - np.mean(delta2_mfccs, axis=0)) / (np.std(delta2_mfccs, axis=0) + 1e-5)
            
            # Combine features
            combined_features = mfccs  # For simplicity, just using base MFCCs
            
        except Exception as e:
            print(f"Error extracting audio with librosa: {str(e)}")
            # Return dummy data if extraction fails
            combined_features = np.zeros((SEQUENCE_LENGTH, 13))
            return combined_features
            
        # Match the sequence length
        if combined_features.shape[0] < SEQUENCE_LENGTH:
            padding = np.zeros((SEQUENCE_LENGTH - combined_features.shape[0], combined_features.shape[1]))
            combined_features = np.vstack((combined_features, padding))
        elif combined_features.shape[0] > SEQUENCE_LENGTH:
            combined_features = combined_features[:SEQUENCE_LENGTH, :]
            
        return combined_features
        
    def predict_with_lipnet(self, video_path):
        """Predict text using only the visual LipNet model on GRID dataset"""
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
        """Predict text using cross-modal attention model on GRID dataset"""
        start_time = time.time()
        
        # Extract mouth frames
        mouth_frames = self.extract_mouth_frames(video_path)
        
        # Add batch dimension
        mouth_frames = np.expand_dims(mouth_frames, axis=0)
        
        # Extract audio features
        audio_features = self.extract_audio_features(video_path)
        
        # Add batch dimension
        audio_features = np.expand_dims(audio_features, axis=0)
        
        # Print shapes for debugging
        print(f"Mouth frames shape: {mouth_frames.shape}")
        print(f"Audio features shape: {audio_features.shape}")
        
        # Predict with cross-modal attention model
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
        """Convert model output to text with improved CTC decoding for GRID dataset"""
        # Using greedy decoding for simplicity
        # In production, implement beam search with language model
        
        # Get the most likely character at each position
        indices = np.argmax(prediction, axis=1)
        
        # Remove consecutive duplicates
        collapsed_indices = []
        for i, idx in enumerate(indices):
            if i == 0 or idx != indices[i-1]:
                collapsed_indices.append(idx)
                
        # Remove blank tokens (index 0)
        filtered_indices = [idx for idx in collapsed_indices if idx != 0]
        
        # Convert indices to characters
        predicted_text = ''.join([NUM_TO_CHAR.get(idx, '') for idx in filtered_indices])
        
        # Post-processing specific to GRID dataset structure
        # This can use the restricted vocabulary to improve accuracy
        predicted_text = self.apply_grid_language_model(predicted_text)
        
        return predicted_text
        
    def apply_grid_language_model(self, raw_text):
        """Apply GRID dataset constraints to improve prediction accuracy"""
        # Convert to lowercase and remove extra spaces
        text = raw_text.lower().strip()
        words = text.split()
        
        if not words:
            return ""
            
        # Apply GRID grammar constraints (command + color + preposition + letter + digit + adverb)
        corrected_words = []
        
        # Try to match the first word to a command
        if words and words[0]:
            best_command = self.find_closest_match(words[0], GRID_COMMANDS)
            corrected_words.append(best_command)
        
        # Try to match the second word to a color
        if len(words) > 1 and words[1]:
            best_color = self.find_closest_match(words[1], GRID_COLORS)
            corrected_words.append(best_color)
        
        # Try to match the third word to a preposition
        if len(words) > 2 and words[2]:
            best_prep = self.find_closest_match(words[2], GRID_PREPOSITIONS)
            corrected_words.append(best_prep)
        
        # Try to match the fourth word to a letter
        if len(words) > 3 and words[3]:
            # For single letters, use direct matching
            if len(words[3]) == 1 and words[3] in GRID_LETTERS:
                corrected_words.append(words[3])
            else:
                best_letter = self.find_closest_match(words[3], GRID_LETTERS)
                corrected_words.append(best_letter)
        
        # Try to match the fifth word to a digit
        if len(words) > 4 and words[4]:
            best_digit = self.find_closest_match(words[4], GRID_DIGITS)
            corrected_words.append(best_digit)
        
        # Try to match the sixth word to an adverb
        if len(words) > 5 and words[5]:
            best_adverb = self.find_closest_match(words[5], GRID_ADVERBS)
            corrected_words.append(best_adverb)
        
        # Join the corrected words
        corrected_text = " ".join(corrected_words)
        
        return corrected_text
        
    def find_closest_match(self, word, word_list):
        """Find the closest matching word from a list based on edit distance"""
        if not word:
            return word_list[0]  # Default to first word if empty
            
        # Calculate edit distances
        distances = [self.levenshtein_distance(word, ref_word) for ref_word in word_list]
        
        # Find the index of the minimum distance
        min_index = distances.index(min(distances))
        
        return word_list[min_index]
        
    def levenshtein_distance(self, s1, s2):
        """Calculate the Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
            
        if len(s2) == 0:
            return len(s1)
            
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
            
        return previous_row[-1]
        
    def evaluate_models(self, video_path, ground_truth_text):
        """Compare performance of visual-only and cross-modal approaches on GRID dataset"""
        # Predict with visual-only LipNet
        visual_result = self.predict_with_lipnet(video_path)
        
        # Predict with cross-modal approach
        multimodal_result = self.predict_with_multimodal(video_path)
        
        # Calculate metrics
        visual_accuracy = self.calculate_word_accuracy(visual_result['text'], ground_truth_text)
        multimodal_accuracy = self.calculate_word_accuracy(multimodal_result['text'], ground_truth_text)
        
        # Calculate character error rate
        visual_cer = self.calculate_cer(visual_result['text'], ground_truth_text)
        multimodal_cer = self.calculate_cer(multimodal_result['text'], ground_truth_text)
        
        # Calculate GRID-specific sentence accuracy (all 6 words correct)
        visual_sentence_acc = 1.0 if visual_result['text'].lower() == ground_truth_text.lower() else 0.0
        multimodal_sentence_acc = 1.0 if multimodal_result['text'].lower() == ground_truth_text.lower() else 0.0
        
        comparison = {
            'visual_only': {
                'predicted_text': visual_result['text'],
                'confidence': visual_result['confidence'],
                'word_accuracy': visual_accuracy,
                'character_error_rate': visual_cer,
                'sentence_accuracy': visual_sentence_acc,
                'processing_time': visual_result['processing_time']
            },
            'cross_modal_attention': {
                'predicted_text': multimodal_result['text'],
                'confidence': multimodal_result['confidence'],
                'word_accuracy': multimodal_accuracy,
                'character_error_rate': multimodal_cer,
                'sentence_accuracy': multimodal_sentence_acc,
                'processing_time': multimodal_result['processing_time']
            },
            'ground_truth': {
                'text': ground_truth_text
            }
        }
        
        return comparison
        
    def calculate_word_accuracy(self, predicted_text, ground_truth):
        """Calculate word-level accuracy optimized for GRID dataset"""
        pred_words = predicted_text.lower().split()
        truth_words = ground_truth.lower().split()
        
        # Count matching words
        correct = 0
        for i in range(min(len(pred_words), len(truth_words))):
            if pred_words[i] == truth_words[i]:
                correct += 1
                
        # Apply penalty for length difference
        total = max(len(pred_words), len(truth_words))
        
        return correct / total if total > 0 else 0
        
    def calculate_cer(self, predicted_text, ground_truth):
        """Calculate character error rate"""
        pred_chars = list(predicted_text.lower())
        truth_chars = list(ground_truth.lower())
        
        # Calculate Levenshtein distance
        distance = self.levenshtein_distance(predicted_text.lower(), ground_truth.lower())
        
        # CER = Levenshtein distance / length of ground truth
        cer = distance / len(truth_chars) if len(truth_chars) > 0 else 0
        
        return cer
        
    def visualize_comparison(self, comparison_results):
        """Visualize the performance comparison between visual-only and cross-modal approaches"""
        metrics = ['Word Accuracy', 'Character Error Rate', 'Sentence Accuracy', 'Processing Time', 'Confidence']
        visual_values = [
            comparison_results['visual_only']['word_accuracy'], 
            comparison_results['visual_only']['character_error_rate'],
            comparison_results['visual_only']['sentence_accuracy'],
            comparison_results['visual_only']['processing_time'],
            comparison_results['visual_only']['confidence']
        ]
        multimodal_values = [
            comparison_results['cross_modal_attention']['word_accuracy'], 
            comparison_results['cross_modal_attention']['character_error_rate'],
            comparison_results['cross_modal_attention']['sentence_accuracy'],
            comparison_results['cross_modal_attention']['processing_time'],
            comparison_results['cross_modal_attention']['confidence']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 8))
        visual_bars = ax.bar(x - width/2, visual_values, width, label='Visual Only (LipNet)')
        multimodal_bars = ax.bar(x + width/2, multimodal_values, width, label='Cross-Modal Attention')
        
        ax.set_ylabel('Value')
        ax.set_title('Performance Comparison: Visual-Only vs Cross-Modal Attention (GRID Dataset)')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        
        # Add exact values above bars
        for i, v in enumerate(visual_values):
            ax.text(i - width/2, v + 0.02, f'{v:.2f}', ha='center')
        
        for i, v in enumerate(multimodal_values):
            ax.text(i + width/2, v + 0.02, f'{v:.2f}', ha='center')
        
        plt.tight_layout()
        plt.savefig('grid_performance_comparison.png')
        plt.show()
        
        # Print the predicted texts and ground truth
        print("-" * 80)
        print("PREDICTION RESULTS:")
        print(f"Visual-Only:        '{comparison_results['visual_only']['predicted_text']}'")
        print(f"Cross-Modal:        '{comparison_results['cross_modal_attention']['predicted_text']}'")
        print(f"Ground Truth:       '{comparison_results['ground_truth']['text']}'")
        print("-" * 80)
        
    def process_grid_dataset(self, dataset_path, num_samples=10):
        """Process a subset of GRID dataset for evaluation"""
        # Get all video files
        video_files = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.mpg'):
                    video_files.append(os.path.join(root, file))
                    
        # Select a random subset
        if len(video_files) > num_samples:
            video_files = random.sample(video_files, num_samples)
            
        results = []
        
        for video_file in video_files:
            # Extract ground truth from filename
            # GRID dataset filenames follow pattern: s{speaker_id}/[align/video]/.../{alignfile}.mpg
            # We need to extract the corresponding align file
            align_file = video_file.replace('.mpg', '.align')
            
            if os.path.exists(align_file):
                with open(align_file, 'r') as f:
                    align_content = f.read()
                    # Extract words from align file
                    words = re.findall(r'\d+ \d+ ([a-z]+)', align_content)
                    ground_truth = ' '.join(words)
            else:
                # If align file not found, try to parse from filename
                basename = os.path.basename(video_file)
                # If align file not found, try to parse from filename
                basename = os.path.basename(video_file)
                # GRID utterances follow a standard structure: command + color + preposition + letter + digit + adverb
                # Try to extract components from filename or use a default structure
                ground_truth = "bin blue at f zero again"  # Default placeholder

            print(f"Processing: {video_file}")
            print(f"Ground truth: {ground_truth}")

            # Evaluate both models
            comparison = self.evaluate_models(video_file, ground_truth)
            results.append(comparison)

            # Visualize individual result
            self.visualize_comparison(comparison)

        # Calculate aggregate metrics
        self.aggregate_results(results)

        return results

    def aggregate_results(self, results_list):
        """Calculate and display aggregate metrics for both models"""
        visual_word_acc = sum(r['visual_only']['word_accuracy'] for r in results_list) / len(results_list)
        multimodal_word_acc = sum(r['cross_modal_attention']['word_accuracy'] for r in results_list) / len(results_list)

        visual_cer = sum(r['visual_only']['character_error_rate'] for r in results_list) / len(results_list)
        multimodal_cer = sum(r['cross_modal_attention']['character_error_rate'] for r in results_list) / len(results_list)

        visual_sent_acc = sum(r['visual_only']['sentence_accuracy'] for r in results_list) / len(results_list)
        multimodal_sent_acc = sum(r['cross_modal_attention']['sentence_accuracy'] for r in results_list) / len(results_list)

        visual_time = sum(r['visual_only']['processing_time'] for r in results_list) / len(results_list)
        multimodal_time = sum(r['cross_modal_attention']['processing_time'] for r in results_list) / len(results_list)

        print("\n" + "=" * 50)
        print("AGGREGATE RESULTS:")
        print("=" * 50)
        print(f"Number of samples: {len(results_list)}")
        print("\nWord Accuracy:")
        print(f"  Visual-Only:     {visual_word_acc:.4f}")
        print(f"  Cross-Modal:     {multimodal_word_acc:.4f}")
        print(f"  Improvement:     {(multimodal_word_acc - visual_word_acc) * 100:.2f}%")

        print("\nCharacter Error Rate:")
        print(f"  Visual-Only:     {visual_cer:.4f}")
        print(f"  Cross-Modal:     {multimodal_cer:.4f}")
        print(f"  Improvement:     {(visual_cer - multimodal_cer) * 100:.2f}%")

        print("\nSentence Accuracy:")
        print(f"  Visual-Only:     {visual_sent_acc:.4f}")
        print(f"  Cross-Modal:     {multimodal_sent_acc:.4f}")
        print(f"  Improvement:     {(multimodal_sent_acc - visual_sent_acc) * 100:.2f}%")

        print("\nAverage Processing Time:")
        print(f"  Visual-Only:     {visual_time:.4f} seconds")
        print(f"  Cross-Modal:     {multimodal_time:.4f} seconds")
        print("=" * 50)

        # Create a summary visualization
        metrics = ['Word Accuracy', 'Character Error Rate', 'Sentence Accuracy']
        visual_values = [visual_word_acc, visual_cer, visual_sent_acc]
        multimodal_values = [multimodal_word_acc, multimodal_cer, multimodal_sent_acc]

        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 8))
        visual_bars = ax.bar(x - width/2, visual_values, width, label='Visual Only (LipNet)')
        multimodal_bars = ax.bar(x + width/2, multimodal_values, width, label='Cross-Modal Attention')

        ax.set_ylabel('Value')
        ax.set_title('Aggregate Performance: Visual-Only vs Cross-Modal Attention (GRID Dataset)')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()

        # Add exact values above bars
        for i, v in enumerate(visual_values):
            ax.text(i - width/2, v + 0.02, f'{v:.4f}', ha='center')

        for i, v in enumerate(multimodal_values):
            ax.text(i + width/2, v + 0.02, f'{v:.4f}', ha='center')

        plt.tight_layout()
        plt.savefig('grid_aggregate_performance.png')
        plt.show()

def train_lipnet_model(lip_reader, training_path, epochs=20, batch_size=32, validation_split=0.2):
    """Train the LipNet model on GRID dataset"""
    # Get list of video files and their corresponding align files
    video_files = []
    align_files = []

    for root, dirs, files in os.walk(training_path):
        for file in files:
            if file.endswith('.mpg'):
                video_path = os.path.join(root, file)
                align_path = video_path.replace('.mpg', '.align')

                if os.path.exists(align_path):
                    video_files.append(video_path)
                    align_files.append(align_path)

    print(f"Found {len(video_files)} videos with alignment files")

    # Create data generator
    def data_generator(video_paths, align_paths, batch_size):
        num_samples = len(video_paths)
        indices = np.arange(num_samples)

        while True:
            # Shuffle at the beginning of each epoch
            np.random.shuffle(indices)

            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_size_actual = len(batch_indices)

                # Initialize batch arrays
                batch_x = np.zeros((batch_size_actual, SEQUENCE_LENGTH, MOUTH_HEIGHT, MOUTH_WIDTH, 3))

                # Maximum length for CTC loss
                max_label_length = 40

                # Initialize label arrays for CTC loss
                batch_y = np.ones((batch_size_actual, max_label_length)) * -1
                input_length = np.zeros((batch_size_actual, 1))
                label_length = np.zeros((batch_size_actual, 1))

                for j, idx in enumerate(batch_indices):
                    video_path = video_paths[idx]
                    align_path = align_paths[idx]

                    # Extract mouth frames
                    frames = lip_reader.extract_mouth_frames(video_path, augment=True)
                    batch_x[j] = frames

                    # Extract text from align file
                    with open(align_path, 'r') as f:
                        align_content = f.read()
                        words = re.findall(r'\d+ \d+ ([a-z]+)', align_content)
                        text = ' '.join(words)

                    # Convert text to sequence of indices
                    label_seq = [CHAR_TO_NUM[c] for c in text if c in CHAR_TO_NUM]

                    # Set label length
                    curr_label_len = len(label_seq)
                    label_length[j] = curr_label_len

                    # Pad label sequence if needed
                    if curr_label_len > max_label_length:
                        label_seq = label_seq[:max_label_length]

                    # Add to batch_y
                    for k, char_idx in enumerate(label_seq):
                        batch_y[j, k] = char_idx

                    # Set input length (output sequence length from CNN/RNN stack)
                    input_length[j] = SEQUENCE_LENGTH // 2  # Approximate based on pooling layers

                inputs = {
                    'input': batch_x,
                    'labels': batch_y,
                    'input_length': input_length,
                    'label_length': label_length
                }

                outputs = {'ctc': np.zeros((batch_size_actual, 1))}  # Dummy output for CTC loss

                yield (inputs, outputs)

    # Create CTC loss function
    def ctc_loss_func(args):
        y_pred, labels, input_length, label_length = args
        # CTC loss
        return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

    # Create actual model for training with CTC loss
    input_data = Input(shape=(SEQUENCE_LENGTH, MOUTH_HEIGHT, MOUTH_WIDTH, 3), name='input')
    labels = Input(name='labels', shape=(None,), dtype='float32')
    input_length = Input(name='input_length', shape=(1,), dtype='int64')
    label_length = Input(name='label_length', shape=(1,), dtype='int64')

    # Use the same architecture as in _create_lipnet_model
    x = TimeDistributed(Conv2D(32, (3, 3), padding='same', activation='relu'))(input_data)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)

    x = TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)

    x = TimeDistributed(Conv2D(96, (3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)

    x = TimeDistributed(Flatten())(x)

    x = Bidirectional(GRU(256, return_sequences=True))(x)
    x = Dropout(0.2)(x)

    x = Bidirectional(GRU(256, return_sequences=True))(x)
    x = Dropout(0.2)(x)

    predictions = Dense(len(VOCAB) + 1, activation='softmax')(x)

    loss_out = Lambda(ctc_loss_func, output_shape=(1,), name='ctc')([predictions, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(learning_rate=0.0001))

    # Create a prediction model (without CTC loss components)
    prediction_model = Model(inputs=input_data, outputs=predictions)

    # Create training and validation generators
    train_size = int(len(video_files) * (1 - validation_split))
    train_videos = video_files[:train_size]
    train_aligns = align_files[:train_size]
    val_videos = video_files[train_size:]
    val_aligns = align_files[train_size:]

    train_generator = data_generator(train_videos, train_aligns, batch_size)
    val_generator = data_generator(val_videos, val_aligns, batch_size)

    # Train the model
    model.fit(
        train_generator,
        steps_per_epoch=len(train_videos) // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=len(val_videos) // batch_size
    )

    # Save the prediction model
    prediction_model.save(LIPNET_MODEL_PATH)

    return prediction_model

def train_multimodal_model(lip_reader, training_path, epochs=20, batch_size=32, validation_split=0.2):
    """Train the cross-modal attention model on GRID dataset"""
    # Get list of video files and their corresponding align files
    video_files = []
    align_files = []

    for root, dirs, files in os.walk(training_path):
        for file in files:
            if file.endswith('.mpg'):
                video_path = os.path.join(root, file)
                align_path = video_path.replace('.mpg', '.align')

                if os.path.exists(align_path):
                    video_files.append(video_path)
                    align_files.append(align_path)

    print(f"Found {len(video_files)} videos with alignment files for multimodal training")

    # Create data generator for multimodal model
    def multimodal_data_generator(video_paths, align_paths, batch_size):
        num_samples = len(video_paths)
        indices = np.arange(num_samples)

        while True:
            # Shuffle at the beginning of each epoch
            np.random.shuffle(indices)

            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_size_actual = len(batch_indices)

                # Initialize batch arrays
                batch_visual = np.zeros((batch_size_actual, SEQUENCE_LENGTH, MOUTH_HEIGHT, MOUTH_WIDTH, 3))
                batch_audio = np.zeros((batch_size_actual, SEQUENCE_LENGTH, 13))  # MFCC features

                # Maximum length for CTC loss
                max_label_length = 40

                # Initialize label arrays for CTC loss
                batch_y = np.ones((batch_size_actual, max_label_length)) * -1
                input_length = np.zeros((batch_size_actual, 1))
                label_length = np.zeros((batch_size_actual, 1))

                for j, idx in enumerate(batch_indices):
                    video_path = video_paths[idx]
                    align_path = align_paths[idx]

                    # Extract mouth frames
                    frames = lip_reader.extract_mouth_frames(video_path, augment=True)
                    batch_visual[j] = frames

                    # Extract audio features
                    audio_features = lip_reader.extract_audio_features(video_path)
                    batch_audio[j] = audio_features

                    # Extract text from align file
                    with open(align_path, 'r') as f:
                        align_content = f.read()
                        words = re.findall(r'\d+ \d+ ([a-z]+)', align_content)
                        text = ' '.join(words)

                    # Convert text to sequence of indices
                    label_seq = [CHAR_TO_NUM[c] for c in text if c in CHAR_TO_NUM]

                    # Set label length
                    curr_label_len = len(label_seq)
                    label_length[j] = curr_label_len

                    # Pad label sequence if needed
                    if curr_label_len > max_label_length:
                        label_seq = label_seq[:max_label_length]

                    # Add to batch_y
                    for k, char_idx in enumerate(label_seq):
                        batch_y[j, k] = char_idx

                    # Set input length (output sequence length from CNN/RNN stack)
                    input_length[j] = SEQUENCE_LENGTH // 2  # Approximate based on pooling layers

                inputs = {
                    'visual_input': batch_visual,
                    'audio_input': batch_audio,
                    'labels': batch_y,
                    'input_length': input_length,
                    'label_length': label_length
                }

                outputs = {'ctc': np.zeros((batch_size_actual, 1))}  # Dummy output for CTC loss

                yield (inputs, outputs)

    # Create CTC loss function
    def ctc_loss_func(args):
        y_pred, labels, input_length, label_length = args
        return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

    # Define model inputs
    visual_input = Input(shape=(SEQUENCE_LENGTH, MOUTH_HEIGHT, MOUTH_WIDTH, 3), name='visual_input')
    audio_input = Input(shape=(SEQUENCE_LENGTH, 13), name='audio_input')
    labels = Input(name='labels', shape=(None,), dtype='float32')
    input_length = Input(name='input_length', shape=(1,), dtype='int64')
    label_length = Input(name='label_length', shape=(1,), dtype='int64')

    # Visual branch
    v = TimeDistributed(Conv2D(32, (3, 3), padding='same', activation='relu'))(visual_input)
    v = TimeDistributed(BatchNormalization())(v)
    v = TimeDistributed(MaxPooling2D((2, 2)))(v)

    v = TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu'))(v)
    v = TimeDistributed(BatchNormalization())(v)
    v = TimeDistributed(MaxPooling2D((2, 2)))(v)

    v = TimeDistributed(Conv2D(96, (3, 3), padding='same', activation='relu'))(v)
    v = TimeDistributed(BatchNormalization())(v)
    v = TimeDistributed(MaxPooling2D((2, 2)))(v)

    v = TimeDistributed(Flatten())(v)

    # Audio branch
    a = Conv1D(128, kernel_size=5, activation='relu', padding='same')(audio_input)
    a = BatchNormalization()(a)
    a = MaxPooling1D(pool_size=2)(a)

    a = Conv1D(256, kernel_size=3, activation='relu', padding='same')(a)
    a = BatchNormalization()(a)
    a = MaxPooling1D(pool_size=2)(a)

    a = Bidirectional(LSTM(128, return_sequences=True))(a)

    # Process audio features to match visual feature dimension
    a_proj = Dense(256, activation='relu')(a)
    a_proj = Dropout(0.2)(a_proj)

    # Process visual features to prepare for attention
    v_proj = Dense(256, activation='relu')(v)
    v_proj = Dropout(0.2)(v_proj)

    # Cross-modal attention
    v2a_attention = MultiHeadAttention(num_heads=8, key_dim=32)(
        query=v_proj,
        key=a_proj,
        value=a_proj
    )
    v2a_attention = Dropout(0.1)(v2a_attention)
    v2a_add = Add()([v_proj, v2a_attention])
    v2a_norm = LayerNormalization()(v2a_add)

    a2v_attention = MultiHeadAttention(num_heads=8, key_dim=32)(
        query=a_proj,
        key=v_proj,
        value=v_proj
    )
    a2v_attention = Dropout(0.1)(a2v_attention)
    a2v_add = Add()([a_proj, a2v_attention])
    a2v_norm = LayerNormalization()(a2v_add)

    # Combine both attention outputs
    cross_modal_features = Concatenate()([v2a_norm, a2v_norm])

    # Final processing
    x = Dense(512, activation='relu')(cross_modal_features)
    x = Dropout(0.3)(x)
    x = Bidirectional(GRU(256, return_sequences=True))(x)
    x = Dropout(0.3)(x)

    # Output layer
    predictions = Dense(len(VOCAB) + 1, activation='softmax')(x)  # +1 for CTC blank

    # Add CTC loss
    loss_out = Lambda(ctc_loss_func, output_shape=(1,), name='ctc')([predictions, labels, input_length, label_length])

    # Create training model with CTC loss
    model = Model(inputs=[visual_input, audio_input, labels, input_length, label_length], outputs=loss_out)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(learning_rate=0.0001))

    # Create prediction model without CTC components
    prediction_model = Model(inputs=[visual_input, audio_input], outputs=predictions)

    # Create training and validation generators
    train_size = int(len(video_files) * (1 - validation_split))
    train_videos = video_files[:train_size]
    train_aligns = align_files[:train_size]
    val_videos = video_files[train_size:]
    val_aligns = align_files[train_size:]

    train_generator = multimodal_data_generator(train_videos, train_aligns, batch_size)
    val_generator = multimodal_data_generator(val_videos, val_aligns, batch_size)

    # Train the model
    model.fit(
        train_generator,
        steps_per_epoch=len(train_videos) // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=len(val_videos) // batch_size
    )

    # Save the prediction model
    prediction_model.save('multimodal_grid_model.h5')

    return prediction_model

def main():
    """Main function to run the lip reading system"""
    print("GRID Dataset Lip Reading System with Cross-Modal Attention")
    print("=" * 60)

    # Initialize the Lip Reader system
    lip_reader = LipReaderSystem(LIPNET_MODEL_PATH)

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Lip Reading System for GRID dataset')
    parser.add_argument('--mode', type=str, default='evaluate',
                        choices=['train', 'evaluate', 'predict', 'demo'],
                        help='Operation mode: train, evaluate, predict, or demo')
    parser.add_argument('--model', type=str, default='multimodal',
                        choices=['lipnet', 'multimodal', 'both'],
                        help='Model to use: lipnet, multimodal, or both')
    parser.add_argument('--input', type=str, default=None,
                        help='Input video file path for prediction mode')
    parser.add_argument('--dataset', type=str, default=GRID_DATASET_PATH,
                        help='Path to GRID dataset')
    parser.add_argument('--samples', type=int, default=10,
                        help='Number of samples to evaluate')

    args = parser.parse_args()

    if args.mode == 'train':
        if args.model == 'lipnet' or args.model == 'both':
            print("Training LipNet model...")
            train_lipnet_model(lip_reader, args.dataset)

        if args.model == 'multimodal' or args.model == 'both':
            print("Training Cross-Modal Attention model...")
            train_multimodal_model(lip_reader, args.dataset)

    elif args.mode == 'evaluate':
        print(f"Evaluating models on {args.samples} samples from GRID dataset...")
        results = lip_reader.process_grid_dataset(args.dataset, args.samples)

    elif args.mode == 'predict':
        if args.input is None:
            print("Error: Input video file is required for predict mode.")
            return

        print(f"Predicting lip reading for video: {args.input}")

        if args.model == 'lipnet' or args.model == 'both':
            result = lip_reader.predict_with_lipnet(args.input)
            print(f"LipNet prediction: {result['text']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Processing time: {result['processing_time']:.4f} seconds")

        if args.model == 'multimodal' or args.model == 'both':
            result = lip_reader.predict_with_multimodal(args.input)
            print(f"Cross-Modal prediction: {result['text']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Processing time: {result['processing_time']:.4f} seconds")

    elif args.mode == 'demo':
        import time
        import cv2

        if args.input is None:
            print("Using webcam as input for demo mode")
            cap = cv2.VideoCapture(0)
        else:
            print(f"Using video file as input: {args.input}")
            cap = cv2.VideoCapture(args.input)

        frames_buffer = []
        processed_results = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Display the frame
            cv2.imshow('Lip Reading Demo', frame)

            # Add frame to buffer
            frames_buffer.append(frame)

            # Keep only the latest SEQUENCE_LENGTH frames
            if len(frames_buffer) > SEQUENCE_LENGTH:
                frames_buffer.pop(0)

            # If we have accumulated enough frames, process them
            if len(frames_buffer) == SEQUENCE_LENGTH:
                # Create a temporary video file
                temp_video = 'temp_demo.mp4'

                height, width = frames_buffer[0].shape[:2]
                writer = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))

                for f in frames_buffer:
                    writer.write(f)

                writer.release()

                # Process with selected model
                if args.model == 'lipnet':
                    result = lip_reader.predict_with_lipnet(temp_video)
                else:  # Default to multimodal
                    result = lip_reader.predict_with_multimodal(temp_video)

                # Keep only the last 5 results for a moving window display
                processed_results.append(result)
                if len(processed_results) > 5:
                    processed_results.pop(0)

                # Display the prediction
                result_text = "Predictions: "
                for idx, res in enumerate(processed_results):
                    result_text += f" [{res['text']}]"

                # Draw the text on screen
                cv2.putText(frame, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
                cv2.imshow('Lip Reading Demo', frame)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Clean up temp file
        if os.path.exists('temp_demo.mp4'):
            os.remove('temp_demo.mp4')

if __name__ == "__main__":
    main()




