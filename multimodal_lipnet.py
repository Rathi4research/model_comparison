import os
import cv2
import numpy as np
import tensorflow as tf
import librosa
import time
from datetime import datetime
import argparse
import dlib

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define paths for models
LIPNET_PATH = "D:\Codebase\model_comparison\pretrained_models\lipnet_weights.h5"
AUDIO_MODEL_PATH = "D:\Codebase\model_comparison\pretrained_models\deepspeech.pb"

class VideoProcessor:
    """Class to handle video pre-processing for lip reading models"""
    
    def __init__(self, target_height=64, target_width=128):
        self.target_height = target_height
        self.target_width = target_width
        
    def extract_mouth_region(self, frame, face_detector, landmark_predictor):
        """Extract mouth region from a frame using dlib"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)
        
        if len(faces) == 0:
            return None
            
        # Get the first face
        face = faces[0]
        landmarks = landmark_predictor(gray, face)
        
        # Extract mouth landmarks (points 48-68 in dlib's 68 point model)
        mouth_points = np.array([[landmarks.part(i).x, landmarks.part(i).y] 
                               for i in range(48, 68)])
        
        # Get bounding box with some margin
        x, y = np.min(mouth_points, axis=0)
        w, h = np.max(mouth_points, axis=0) - np.min(mouth_points, axis=0)
        
        # Add margin
        margin = int(max(w, h) * 0.3)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w += 2 * margin
        h += 2 * margin
        
        # Ensure we don't go beyond frame boundaries
        h_frame, w_frame = frame.shape[:2]
        w = min(w_frame - x, w)
        h = min(h_frame - y, h)
        
        # Extract and resize mouth region
        mouth_roi = frame[y:y+h, x:x+w]
        mouth_roi = cv2.resize(mouth_roi, (self.target_width, self.target_height))
        
        return mouth_roi
    
    def process_video(self, video_path, face_detector=None, landmark_predictor=None):
        """Process video for lip reading, extracting mouth sequences"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # Load dlib if not provided
        if face_detector is None or landmark_predictor is None:
            face_detector = dlib.get_frontal_face_detector()
            landmark_predictor = dlib.shape_predictor("D:\Codebase\model_comparison\pretrained_models\shape_predictor_68_face_landmarks.dat")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            mouth_roi = self.extract_mouth_region(frame, face_detector, landmark_predictor)
            if mouth_roi is not None:
                frames.append(mouth_roi)
        
        cap.release()
        
        # Convert to required format
        frames = np.array(frames)
        return frames

class AudioProcessor:
    """Class to handle audio extraction and preprocessing"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
    def extract_audio(self, video_path):
        """Extract audio from video file using librosa"""
        try:
            # First check if the video has audio
            cap = cv2.VideoCapture(video_path)
            has_audio = cap.get(cv2.CAP_PROP_AUDIO_STREAM) > 0
            cap.release()
            
            if not has_audio:
                print(f"No audio stream found in {video_path}")
                return None
                
            # Extract audio using librosa
            audio_temp_path = os.path.splitext(video_path)[0] + "_temp_audio.wav"
            
            # Use ffmpeg to extract audio
            os.system(f"ffmpeg -y -i {video_path} -ac 1 -ar {self.sample_rate} -loglevel error {audio_temp_path}")
            
            # Load audio with librosa
            audio, _ = librosa.load(audio_temp_path, sr=self.sample_rate)
            
            # Remove temporary file
            if os.path.exists(audio_temp_path):
                os.remove(audio_temp_path)
                
            return audio
            
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return None
    
    def preprocess_audio(self, audio):
        """Preprocess audio for the speech recognition model"""
        if audio is None:
            return None
            
        # Normalize audio
        audio = audio / (np.max(np.abs(audio)) + 1e-10)
        
        # Apply pre-emphasis filter
        pre_emphasis = 0.97
        emphasized_audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
        
        return emphasized_audio

class LipNetModel:
    """Wrapper for LipNet model"""
    
    def __init__(self, model_path=LIPNET_PATH):
        self.model_path = model_path
        self.model = None
        self.char_to_idx = None
        self.idx_to_char = None
        self.load_model()
        
    def load_model(self):
        """Load pre-trained LipNet model"""
        # Define character mapping (simplified example - actual might be more complex)
        chars = "abcdefghijklmnopqrstuvwxyz' "
        self.char_to_idx = {char: idx for idx, char in enumerate(chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(chars)}
        
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print("LipNet model loaded successfully")
        except Exception as e:
            print(f"Error loading LipNet model: {e}")
            # Create placeholder model for demo purposes
            input_shape = (75, 64, 128, 3)  # Example dimensions
            self.model = self._create_lipnet_placeholder(input_shape, len(chars))
            print("Created placeholder LipNet model")
    
    def _create_lipnet_placeholder(self, input_shape, num_classes):
        """Create a simplified LipNet architecture as placeholder"""
        inputs = tf.keras.Input(shape=input_shape)
        
        # Simplified convolutional frontend
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), padding='same'))(inputs)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Activation('relu'))(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2)))(x)
        
        # More conv layers
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), padding='same'))(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Activation('relu'))(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2)))(x)
        
        # Reshape for RNN
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
        
        # RNN layers
        x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True))(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True))(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(num_classes + 1, activation='softmax')(x)  # +1 for CTC blank
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss=self._ctc_loss_placeholder)
        
        return model
    
    def _ctc_loss_placeholder(self, y_true, y_pred):
        """Placeholder for CTC loss"""
        return tf.reduce_mean(y_pred - y_true)
    
    def preprocess_frames(self, frames):
        """Preprocess frames for LipNet model"""
        # Normalize frames
        frames = frames.astype(np.float32) / 255.0
        
        # Ensure we have exact frame count needed by model
        target_frames = 75  # LipNet typically uses 75 frames
        
        if len(frames) < target_frames:
            # Pad with zeros if too short
            padding = np.zeros((target_frames - len(frames), *frames.shape[1:]), dtype=frames.dtype)
            frames = np.vstack([frames, padding])
        elif len(frames) > target_frames:
            # Truncate if too long
            frames = frames[:target_frames]
            
        return np.expand_dims(frames, axis=0)  # Add batch dimension
    
    def decode_prediction(self, prediction):
        """Decode prediction from model output"""
        # In actual implementation, you'd use CTC beam search decoding
        # This is a simplified example for demonstration
        
        # Get most likely character at each timestep
        pred_indices = np.argmax(prediction[0], axis=1)
        
        # Merge repeated characters and remove blanks (CTC decoding simplified)
        text = ""
        prev_idx = -1
        for idx in pred_indices:
            if idx != prev_idx and idx < len(self.idx_to_char):  # Not blank and not repeated
                text += self.idx_to_char[idx]
            prev_idx = idx
            
        return text
    
    def predict(self, frames):
        """Make prediction from video frames"""
        start_time = time.time()
        
        # Preprocess frames
        processed_frames = self.preprocess_frames(frames)
        
        # Make prediction
        prediction = self.model.predict(processed_frames)
        
        # Decode prediction
        text = self.decode_prediction(prediction)
        
        end_time = time.time()
        
        return {
            "text": text,
            "confidence": np.max(prediction),  # Simplified confidence score
            "processing_time": end_time - start_time
        }

class AudioModel:
    """Wrapper for speech recognition model (e.g., DeepSpeech)"""
    
    def __init__(self, model_path=AUDIO_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load pre-trained speech recognition model"""
        try:
            # In a real implementation, you'd load the DeepSpeech model
            # For simplicity, we're using a placeholder
            print("Would load DeepSpeech model from:", self.model_path)
            self.model = "DeepSpeech placeholder"
            print("Audio model ready (placeholder)")
        except Exception as e:
            print(f"Error loading audio model: {e}")
    
    def predict(self, audio):
        """Make prediction from audio data"""
        if audio is None:
            return {"text": "", "confidence": 0.0, "processing_time": 0.0}
            
        start_time = time.time()
        
        # In a real implementation, you'd run inference with DeepSpeech
        # For demonstration, we'll simulate a prediction
        # This would be replaced with actual model inference
        if isinstance(audio, np.ndarray) and len(audio) > 0:
            # Simulate a prediction based on audio length
            # In real implementation, you'd use the actual model
            text_options = [
                "hello world",
                "how are you today",
                "deep learning is fascinating", 
                "speech recognition combined with lip reading",
                "this is a multimodal approach to speech recognition"
            ]
            # Select a "prediction" based on audio length
            text = text_options[len(audio) % len(text_options)]
            confidence = 0.8  # Simulated confidence
        else:
            text = ""
            confidence = 0.0
            
        end_time = time.time()
        
        return {
            "text": text,
            "confidence": confidence,
            "processing_time": end_time - start_time
        }

class MultiModalLipReader:
    """Multimodal lip reading system that combines visual and audio cues"""
    
    def __init__(self):
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor()
        self.lip_model = LipNetModel()
        self.audio_model = AudioModel()
        
    def predict(self, video_path, fusion_method='weighted'):
        """
        Make prediction using both visual and audio cues when available
        
        Args:
            video_path: Path to the video file
            fusion_method: Method to combine predictions ('weighted', 'confidence', or 'best')
                - 'weighted': Use a weighted combination of both predictions
                - 'confidence': Use the prediction with highest confidence
                - 'best': Use audio if available, fallback to visual
        
        Returns:
            Dictionary containing prediction results
        """
        start_time = time.time()
        result = {}
        
        # Process video for lip reading
        print("Processing video frames...")
        frames = self.video_processor.process_video(video_path)
        
        # Extract audio if available
        print("Processing audio...")
        audio = self.audio_processor.extract_audio(video_path)
        processed_audio = self.audio_processor.preprocess_audio(audio)
        
        # Get predictions from both modalities
        lip_result = self.lip_model.predict(frames)
        print(f"LipNet prediction: '{lip_result['text']}'")
        
        has_audio = processed_audio is not None and len(processed_audio) > 0
        if has_audio:
            audio_result = self.audio_model.predict(processed_audio)
            print(f"Audio model prediction: '{audio_result['text']}'")
        else:
            audio_result = {"text": "", "confidence": 0.0, "processing_time": 0.0}
            print("No audio available or audio processing failed")
        
        # Store individual results
        result["lip_only"] = lip_result
        result["audio_only"] = audio_result
        
        # Fuse predictions based on the chosen method
        if has_audio:
            fusion_start = time.time()
            
            if fusion_method == 'weighted':
                # Weighted combination based on confidence scores
                total_conf = lip_result["confidence"] + audio_result["confidence"]
                if total_conf > 0:
                    lip_weight = lip_result["confidence"] / total_conf
                    audio_weight = audio_result["confidence"] / total_conf
                else:
                    lip_weight = audio_weight = 0.5
                
                # Use simple text fusion (in practice, would use more sophisticated methods)
                if lip_weight > audio_weight * 2:  # Lip reading much more confident
                    text = lip_result["text"]
                    confidence = lip_result["confidence"]
                elif audio_weight > lip_weight * 2:  # Audio much more confident
                    text = audio_result["text"]
                    confidence = audio_result["confidence"]
                else:
                    # Simple fusion: choose longer text if both seem equally valid
                    # In a real implementation, you might use word-level combination
                    if len(lip_result["text"]) > len(audio_result["text"]):
                        text = lip_result["text"]
                    else:
                        text = audio_result["text"]
                    confidence = (lip_result["confidence"] + audio_result["confidence"]) / 2
            
            elif fusion_method == 'confidence':
                # Choose the prediction with higher confidence
                if lip_result["confidence"] > audio_result["confidence"]:
                    text = lip_result["text"]
                    confidence = lip_result["confidence"]
                else:
                    text = audio_result["text"]
                    confidence = audio_result["confidence"]
                    
            elif fusion_method == 'best':
                # Use audio if available with reasonable confidence, otherwise lip reading
                if audio_result["confidence"] > 0.4:  # Adjust threshold as needed
                    text = audio_result["text"]
                    confidence = audio_result["confidence"]
                else:
                    text = lip_result["text"]
                    confidence = lip_result["confidence"]
            
            else:  # Default to best method
                if audio_result["confidence"] > 0.4:
                    text = audio_result["text"]
                    confidence = audio_result["confidence"]
                else:
                    text = lip_result["text"]
                    confidence = lip_result["confidence"]
                    
            fusion_time = time.time() - fusion_start
            
        else:
            # No audio, just use lip reading
            text = lip_result["text"]
            confidence = lip_result["confidence"]
            fusion_time = 0.0
        
        end_time = time.time()
        
        # Create final fused result
        result["fused"] = {
            "text": text,
            "confidence": confidence,
            "processing_time": end_time - start_time,
            "fusion_time": fusion_time,
            "fusion_method": fusion_method
        }
        
        return result
    
    def evaluate(self, video_path, ground_truth, fusion_method='weighted'):
        """
        Evaluate the multimodal system against ground truth
        
        Args:
            video_path: Path to the video file
            ground_truth: Actual text spoken in the video
            fusion_method: Method to combine predictions
            
        Returns:
            Evaluation metrics for all modalities
        """
        import Levenshtein
        
        result = self.predict(video_path, fusion_method)
        evaluation = {}
        
        # Calculate metrics for all prediction types
        for key in ["lip_only", "audio_only", "fused"]:
            if key in result:
                predicted = result[key]["text"].lower()
                actual = ground_truth.lower()
                
                # Character error rate (CER)
                cer = Levenshtein.distance(predicted, actual) / max(len(actual), 1)
                
                # Word error rate (WER)
                pred_words = predicted.split()
                actual_words = actual.split()
                wer = Levenshtein.distance(pred_words, actual_words) / max(len(actual_words), 1)
                
                # Add metrics to this prediction type
                result[key]["metrics"] = {
                    "CER": cer,
                    "WER": wer,
                    "ground_truth": ground_truth
                }
                
                # Store metrics separately for reporting
                evaluation[key] = {
                    "CER": cer,
                    "WER": wer,
                    "predicted": predicted,
                    "ground_truth": ground_truth
                }
        
        return result, evaluation
    
    def batch_evaluate(self, video_dir, ground_truth_file, fusion_method='weighted'):
        """
        Evaluate the multimodal system on a batch of videos
        
        Args:
            video_dir: Directory containing video files
            ground_truth_file: CSV file with video_name,text format
            fusion_method: Method to combine predictions
            
        Returns:
            Aggregate evaluation metrics and detailed results
        """
        # Load ground truth
        ground_truth = {}
        if os.path.exists(ground_truth_file):
            with open(ground_truth_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        video_name = os.path.basename(parts[0])
                        text = parts[1]
                        ground_truth[video_name] = text
        
        # Get video files
        video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) 
                      if f.endswith(('.mp4', '.avi', '.mov'))]
        
        # Prepare result containers
        all_results = []
        metrics = {
            "lip_only": {"CER": [], "WER": [], "processing_time": []},
            "audio_only": {"CER": [], "WER": [], "processing_time": []},
            "fused": {"CER": [], "WER": [], "processing_time": []}
        }
        
        # Process each video
        for video_path in video_files:
            video_name = os.path.basename(video_path)
            
            if video_name in ground_truth:
                print(f"\nProcessing {video_name}...")
                result, eval_metrics = self.evaluate(
                    video_path, 
                    ground_truth[video_name],
                    fusion_method
                )
                
                # Store results
                all_results.append({
                    "video_path": video_path,
                    "results": result
                })
                
                # Accumulate metrics
                for modality in ["lip_only", "audio_only", "fused"]:
                    if modality in result and "metrics" in result[modality]:
                        metrics[modality]["CER"].append(result[modality]["metrics"]["CER"])
                        metrics[modality]["WER"].append(result[modality]["metrics"]["WER"])
                        metrics[modality]["processing_time"].append(result[modality]["processing_time"])
                
                # Print individual results
                print(f"\nResults for {video_name}:")
                print(f"Ground truth: '{ground_truth[video_name]}'")
                print(f"Lip only: '{result['lip_only']['text']}' (CER: {result['lip_only']['metrics']['CER']:.4f})")
                
                if result['audio_only']['confidence'] > 0:
                    print(f"Audio only: '{result['audio_only']['text']}' (CER: {result['audio_only']['metrics']['CER']:.4f})")
                else:
                    print("Audio only: No audio available")
                    
                print(f"Fused ({fusion_method}): '{result['fused']['text']}' (CER: {result['fused']['metrics']['CER']:.4f})")
        
        # Calculate summary metrics
        summary = {}
        for modality in metrics:
            if metrics[modality]["CER"]:
                summary[modality] = {
                    "avg_CER": np.mean(metrics[modality]["CER"]),
                    "avg_WER": np.mean(metrics[modality]["WER"]),
                    "avg_processing_time": np.mean(metrics[modality]["processing_time"])
                }
        
        # Print summary
        print("\n" + "="*50)
        print("Summary Results")
        print("="*50)
        
        for modality in ["lip_only", "audio_only", "fused"]:
            if modality in summary:
                print(f"\n{modality.upper()}:")
                print(f"  Average CER: {summary[modality]['avg_CER']:.4f}")
                print(f"  Average WER: {summary[modality]['avg_WER']:.4f}")
                print(f"  Average processing time: {summary[modality]['avg_processing_time']:.4f} seconds")
        
        return summary, all_results
    
    def visualize_results(self, summary, output_dir="multimodal_results"):
        """
        Visualize comparison results
        
        Args:
            summary: Summary metrics from batch_evaluate
            output_dir: Directory to save visualizations
        """
        import matplotlib.pyplot as plt
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Plot metrics comparison
        modalities = []
        cer_values = []
        wer_values = []
        time_values = []
        
        for modality in ["lip_only", "audio_only", "fused"]:
            if modality in summary:
                modalities.append(modality)
                cer_values.append(summary[modality]["avg_CER"])
                wer_values.append(summary[modality]["avg_WER"])
                time_values.append(summary[modality]["avg_processing_time"])
        
        # Plot error rates
        plt.figure(figsize=(10, 6))
        x = np.arange(len(modalities))
        width = 0.35
        
        plt.bar(x - width/2, cer_values, width, label='CER')
        plt.bar(x + width/2, wer_values, width, label='WER')
        
        plt.ylabel('Error Rate')
        plt.title('Average Error Rates by Modality')
        plt.xticks(x, [m.replace('_', ' ').title() for m in modalities])
        plt.legend()
        
        plt.savefig(os.path.join(output_dir, 'error_rates_comparison.png'))
        plt.close()
        
        # Plot processing time
        plt.figure(figsize=(8, 6))
        plt.bar([m.replace('_', ' ').title() for m in modalities], time_values)
        plt.ylabel('Time (seconds)')
        plt.title('Average Processing Time by Modality')
        
        plt.savefig(os.path.join(output_dir, 'processing_time_comparison.png'))
        plt.close()
        
        print(f"Visualizations saved to {output_dir} directory")

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Multimodal Lip Reading System')
    parser.add_argument('--video', type=str, help='Path to a single video file to process')
    parser.add_argument('--video_dir', type=str, help='Path to directory containing multiple videos')
    parser.add_argument('--ground_truth', type=str, help='Path to ground truth text file (video_name,text format)')
    parser.add_argument('--fusion', type=str, default='weighted', choices=['weighted', 'confidence', 'best'],
                       help='Method to fuse visual and audio predictions')
    parser.add_argument('--output_dir', type=str, default='multimodal_results', 
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Initialize the multimodal system
    multimodal_system = MultiModalLipReader()
    
    if args.video:
        # Process single video
        print(f"Processing video: {args.video}")
        if args.ground_truth and os.path.exists(args.ground_truth):
            # Single video with ground truth
            video_name = os.path.basename(args.video)
            ground_truth = None
            
            # Try to find matching ground truth
            with open(args.ground_truth, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2 and os.path.basename(parts[0]) == video_name:
                        ground_truth = parts[1]
                        break
            
            if ground_truth:
                # Evaluate with ground truth
                result, evaluation = multimodal_system.evaluate(args.video, ground_truth, args.fusion)
                
                # Visualize for single video
                single_summary = {
                    modality: {"avg_CER": eval_data["CER"], 
                              "avg_WER": eval_data["WER"],
                              "avg_processing_time": result[modality]["processing_time"]}
                    for modality, eval_data in evaluation.items()
                }
                multimodal_system.visualize_results(single_summary, args.output_dir)
            else:
                # Just predict without evaluation
                result = multimodal_system.predict(args.video, args.fusion)
                print("\nPrediction results:")
                print(f"LipNet: '{result['lip_only']['text']}'")
                if result['audio_only']['confidence'] > 0:
                    print(f"Audio: '{result['audio_only']['text']}'")
                else:
                    print("Audio: No audio available")
                print(f"Fused ({args.fusion}): '{result['fused']['text']}'")
        else:
            # Just predict without evaluation
            result = multimodal_system.predict(args.video, args.fusion)
            print("\nPrediction results:")
            print(f"LipNet: '{result['lip_only']['text']}'")
            if result['audio_only']['confidence'] > 0:
                print(f"Audio: '{result['audio_only']['text']}'")
            else:
                print("Audio: No audio available")
            print(f"Fused ({args.fusion}): '{result['fused']['text']}'")
    
    elif args.video_dir:
        # Process batch of videos
        if not args.ground_truth:
            print("Ground truth file is required for batch processing")
            return
            
        print(f"Processing videos from: {args.video_dir}")
        print(f"Using ground truth from: {args.ground_truth}")
        print(f"Fusion method: {args.fusion}")
        
        summary, all_results = multimodal_system.batch_evaluate(
            args.video_dir, args.ground_truth, args.fusion
        )
        
        # Generate visualizations
        multimodal_system.visualize_results(summary, args.output_dir)
    
    else:
        print("Please provide either --video or --video_dir argument")
        parser.print_help()

if __name__ == "__main__":
    main()
