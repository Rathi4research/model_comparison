import os
import cv2
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
import argparse
from datetime import datetime

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define paths for models (you'll need to download these pre-trained models)
LIPNET_PATH = "D:\Codebase\model_comparison\pretrained_models\lipnet_weights.h5"
WAV2LIP_PATH = "D:\Codebase\model_comparison\pretrained_models\wav2lip.pth"

class VideoProcessor:
    """Class to handle video pre-processing for lip reading models"""

    def __init__(self, target_height=64, target_width=128):
        self.target_height = target_height
        self.target_width = target_width

    def extract_mouth_region(self, frame, face_detector, landmark_predictor):
        """Extract mouth region from a frame using dlib"""
        import dlib  # Import here to make it optional

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
        h, w_frame = frame.shape[:2]
        w = min(w_frame - x, w)
        h = min(h - y, h)

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
            import dlib
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

        # Convert to required format based on the intended model
        frames = np.array(frames)
        return frames

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
        # This is a placeholder - you'd need the actual character mapping used for training
        chars = "abcdefghijklmnopqrstuvwxyz' "
        self.char_to_idx = {char: idx for idx, char in enumerate(chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(chars)}

        # Create a simplified LipNet model structure (you'd replace with actual architecture)
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print("LipNet model loaded successfully")
        except Exception as e:
            print(f"Error loading LipNet model: {e}")
            # Create placeholder model for demo purposes
            # In real implementation, you'd use the actual LipNet architecture
            input_shape = (75, 64, 128, 3)  # Example: 75 frames, 64x128 resolution, 3 channels
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

        # More conv layers would go here in real implementation
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
        """Placeholder for CTC loss - in real implementation use actual CTC loss"""
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

        predicted = text.lower()
        actual = 'about'

        print(f"Actual word: {actual}")
        print(f"Predicted word: {predicted}")

        # Word-level accuracy (simplified)
        pred_words = predicted.split()
        actual_words = actual.split()

        # Character error rate (CER)
        import Levenshtein
        cer = Levenshtein.distance(predicted, actual) / max(len(actual), 1)

        # Word error rate (WER)
        wer = Levenshtein.distance(pred_words, actual_words) / max(len(actual_words), 1)
        print(f"CER: {cer}")
        print(f"WER: {wer}")
        end_time = time.time()

        return {
            "text": text,
            "confidence": np.max(prediction),  # Simplified confidence score
            "processing_time": end_time - start_time,
            "metrics" : {
                "CER": cer,
                "WER": wer
            }
        }

class Wav2LipModel:
    """Wrapper for Wav2Lip model adapted for lip reading"""

    def __init__(self, model_path=WAV2LIP_PATH):
        self.model_path = model_path
        self.model = None
        self.vocab = None
        self.load_model()

    def load_model(self):
        """Load pre-trained Wav2Lip model"""
        # Create vocabulary (simplified example)
        # This is a placeholder - you'd need the actual vocabulary used for training
        chars = "abcdefghijklmnopqrstuvwxyz' "
        self.vocab = {idx: char for idx, char in enumerate(chars)}

        # Create a simplified Wav2Lip model structure
        try:
            self.model = torch.load(self.model_path, map_location='cpu')
            print("Wav2Lip model loaded successfully")
        except Exception as e:
            print(f"Error loading Wav2Lip model: {e}")
            # Create placeholder model for demo purposes
            self.model = self._create_wav2lip_placeholder(len(chars))
            print("Created placeholder Wav2Lip model")

    def _create_wav2lip_placeholder(self, vocab_size):
        """Create a simplified Wav2Lip model as placeholder"""
        class Wav2LipPlaceholder(nn.Module):
            def __init__(self, vocab_size):
                super(Wav2LipPlaceholder, self).__init__()
                # Visual feature extraction
                self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
                self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))

                # RNN layers
                self.gru1 = nn.GRU(64 * 16 * 32, 256, bidirectional=True, batch_first=True)
                self.gru2 = nn.GRU(512, 256, bidirectional=True, batch_first=True)

                # Output layer
                self.fc = nn.Linear(512, vocab_size)

            def forward(self, x):
                batch_size, seq_len, height, width, channels = x.shape
                # Reshape for 3D convolution
                x = x.permute(0, 4, 1, 2, 3)  # [batch, channels, seq_len, height, width]

                # Feature extraction
                x = self.conv1(x)
                x = nn.functional.relu(x)
                x = self.pool1(x)
                x = self.conv2(x)
                x = nn.functional.relu(x)
                x = self.pool2(x)

                # Reshape for RNN
                x = x.permute(0, 2, 1, 3, 4)  # [batch, seq_len, channels, height, width]
                x = x.reshape(batch_size, seq_len, -1)  # [batch, seq_len, features]

                # RNN layers
                x, _ = self.gru1(x)
                x, _ = self.gru2(x)

                # Output layer
                x = self.fc(x)
                x = nn.functional.log_softmax(x, dim=2)

                return x

        return Wav2LipPlaceholder(vocab_size)

    def preprocess_frames(self, frames):
        """Preprocess frames for Wav2Lip model"""
        # Convert to PyTorch tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        processed_frames = []
        for frame in frames:
            # Apply transformations
            frame_tensor = transform(frame)
            processed_frames.append(frame_tensor)

        # Stack frames
        frames_tensor = torch.stack(processed_frames)

        # Add batch dimension
        frames_tensor = frames_tensor.unsqueeze(0)

        # Reshape to model's expected format
        frames_tensor = frames_tensor.permute(0, 1, 3, 4, 2)  # [batch, seq_len, height, width, channels]

        return frames_tensor

    def decode_prediction(self, prediction):
        """Decode prediction from model output"""
        # In actual implementation, you'd use CTC beam search decoding
        # This is a simplified example for demonstration

        # Get most likely character at each timestep
        pred_indices = torch.argmax(prediction[0], dim=1).cpu().numpy()

        # Merge repeated characters and remove blanks (CTC decoding simplified)
        text = ""
        prev_idx = -1
        for idx in pred_indices:
            if idx != prev_idx and idx < len(self.vocab):  # Not blank and not repeated
                text += self.vocab[idx]
            prev_idx = idx

        return text

    def predict(self, frames):
        """Make prediction from video frames"""
        start_time = time.time()

        # Preprocess frames
        self.model.eval()  # Set to evaluation mode
        with torch.no_grad():
            # Preprocess frames
            processed_frames = self.preprocess_frames(frames)

            # Make prediction
            prediction = self.model(processed_frames)

            # Decode prediction
            text = self.decode_prediction(prediction)

        end_time = time.time()

        return {
            "text": text,
            "confidence": torch.max(prediction).item(),  # Simplified confidence score
            "processing_time": end_time - start_time
        }

class LipReadingComparison:
    """Compare different lip reading models on the same videos"""

    def __init__(self):
        self.video_processor = VideoProcessor()
        self.lipnet_model = LipNetModel()
        self.wav2lip_model = Wav2LipModel()

    def process_video(self, video_path):
        """Process a video through both models and compare results"""
        print(f"Processing video: {video_path}")

        # Extract frames
        try:
            frames = self.video_processor.process_video(video_path)
            if len(frames) == 0:
                print("No valid frames extracted from video.")
                return None

            print(f"Extracted {len(frames)} frames from video.")

            # Get predictions from both models
            lipnet_result = self.lipnet_model.predict(frames)
            # wav2lip_result = self.wav2lip_model.predict(frames)

            results = {
                "video_path": video_path,
                "lipnet": lipnet_result,
                # "wav2lip": wav2lip_result,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            # self._print_results(results)
            return results

        except Exception as e:
            print(f"Error processing video: {e}")
            return None

    def process_batch(self, video_dir, ground_truth_file=None):
        """Process a batch of videos and optionally compare with ground truth"""
        video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir)
                       if f.endswith(('.mp4', '.avi', '.mov'))]

        ground_truth = {}
        if ground_truth_file and os.path.exists(ground_truth_file):
            with open(ground_truth_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        video_name = os.path.basename(parts[0])
                        text = parts[1]
                        ground_truth[video_name] = text

        results = []
        metrics = {"lipnet": {}, "wav2lip": {}}

        for video_path in video_files:
            result = self.process_video(video_path)
            if result:
                video_name = os.path.basename(video_path)

                # Add ground truth if available
                if video_name in ground_truth:
                    result["ground_truth"] = ground_truth[video_name]

                    # Calculate metrics (simplified)
                    for model_name in ["lipnet", "wav2lip"]:
                        predicted = result[model_name]["text"].lower()
                        actual = ground_truth[video_name].lower()

                        # Word-level accuracy (simplified)
                        pred_words = predicted.split()
                        actual_words = actual.split()

                        # Character error rate (CER)
                        import Levenshtein
                        cer = Levenshtein.distance(predicted, actual) / max(len(actual), 1)

                        # Word error rate (WER)
                        wer = Levenshtein.distance(pred_words, actual_words) / max(len(actual_words), 1)

                        result[model_name]["metrics"] = {
                            "CER": cer,
                            "WER": wer
                        }

                        # Accumulate metrics for overall comparison
                        if "CER" not in metrics[model_name]:
                            metrics[model_name]["CER"] = []
                            metrics[model_name]["WER"] = []
                            metrics[model_name]["processing_time"] = []

                        metrics[model_name]["CER"].append(cer)
                        metrics[model_name]["WER"].append(wer)
                        metrics[model_name]["processing_time"].append(result[model_name]["processing_time"])

                results.append(result)

        # Calculate aggregate metrics
        summary = {"lipnet": {}, "wav2lip": {}}
        for model_name in ["lipnet", "wav2lip"]:
            if metrics[model_name] and "CER" in metrics[model_name] and len(metrics[model_name]["CER"]) > 0:
                summary[model_name] = {
                    "avg_CER": np.mean(metrics[model_name]["CER"]),
                    "avg_WER": np.mean(metrics[model_name]["WER"]),
                    "avg_processing_time": np.mean(metrics[model_name]["processing_time"])
                }

        self._print_comparison_summary(summary)
        return results, summary

    def _print_results(self, results):
        """Print individual result details"""
        print("\n" + "="*50)
        print(f"Results for video: {os.path.basename(results['video_path'])}")
        print("="*50)

        print("\nLipNet Result:")
        print(f"  Predicted Text: '{results['lipnet']['text']}'")
        print(f"  Confidence: {results['lipnet']['confidence']:.4f}")
        print(f"  Processing Time: {results['lipnet']['processing_time']:.4f} seconds")

        print("\nWav2Lip Result:")
        print(f"  Predicted Text: '{results['wav2lip']['text']}'")
        print(f"  Confidence: {results['wav2lip']['confidence']:.4f}")
        print(f"  Processing Time: {results['wav2lip']['processing_time']:.4f} seconds")

        if "ground_truth" in results:
            print(f"\nGround Truth: '{results['ground_truth']}'")

            if "metrics" in results["lipnet"]:
                print("\nLipNet Metrics:")
                print(f"  Character Error Rate (CER): {results['lipnet']['metrics']['CER']:.4f}")
                print(f"  Word Error Rate (WER): {results['lipnet']['metrics']['WER']:.4f}")

            if "metrics" in results["wav2lip"]:
                print("\nWav2Lip Metrics:")
                print(f"  Character Error Rate (CER): {results['wav2lip']['metrics']['CER']:.4f}")
                print(f"  Word Error Rate (WER): {results['wav2lip']['metrics']['WER']:.4f}")

    def _print_comparison_summary(self, summary):
        """Print summary comparison between models"""
        print("\n" + "="*50)
        print("Overall Comparison Summary")
        print("="*50)

        for model_name, metrics in summary.items():
            if metrics:
                print(f"\n{model_name.upper()} Summary:")
                print(f"  Average Character Error Rate (CER): {metrics['avg_CER']:.4f}")
                print(f"  Average Word Error Rate (WER): {metrics['avg_WER']:.4f}")
                print(f"  Average Processing Time: {metrics['avg_processing_time']:.4f} seconds")

        # Determine which model performed better
        if summary["lipnet"] and summary["wav2lip"]:
            if summary["lipnet"]["avg_CER"] < summary["wav2lip"]["avg_CER"]:
                print("\nLipNet performed better in terms of accuracy (lower CER).")
            elif summary["wav2lip"]["avg_CER"] < summary["lipnet"]["avg_CER"]:
                print("\nWav2Lip performed better in terms of accuracy (lower CER).")
            else:
                print("\nBoth models had similar accuracy performance.")

            if summary["lipnet"]["avg_processing_time"] < summary["wav2lip"]["avg_processing_time"]:
                print("LipNet was faster in processing time.")
            elif summary["wav2lip"]["avg_processing_time"] < summary["lipnet"]["avg_processing_time"]:
                print("Wav2Lip was faster in processing time.")
            else:
                print("Both models had similar processing times.")

    def visualize_comparison(self, results, summary, output_dir="comparison_results"):
        """Generate visualizations for model comparison"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Extract data for plotting
        model_names = ["LipNet", "Wav2Lip"]

        # Plot average metrics
        if summary["lipnet"] and summary["wav2lip"]:
            # CER and WER comparison
            metrics = ["avg_CER", "avg_WER"]
            values = [
                [summary["lipnet"]["avg_CER"], summary["lipnet"]["avg_WER"]],
                [summary["wav2lip"]["avg_CER"], summary["wav2lip"]["avg_WER"]]
            ]

            plt.figure(figsize=(10, 6))
            x = np.arange(len(metrics))
            width = 0.35

            plt.bar(x - width/2, values[0], width, label='LipNet')
            plt.bar(x + width/2, values[1], width, label='Wav2Lip')

            plt.ylabel('Error Rate')
            plt.title('Average Error Rates by Model')
            plt.xticks(x, metrics)
            plt.legend()

            plt.savefig(os.path.join(output_dir, 'error_rates_comparison.png'))
            plt.close()

            # Processing time comparison
            proc_times = [summary["lipnet"]["avg_processing_time"],
                          summary["wav2lip"]["avg_processing_time"]]

            plt.figure(figsize=(8, 6))
            plt.bar(model_names, proc_times)
            plt.ylabel('Time (seconds)')
            plt.title('Average Processing Time by Model')

            plt.savefig(os.path.join(output_dir, 'processing_time_comparison.png'))
            plt.close()

        # If we have individual results with ground truth, plot per-video CER
        video_cers = {"LipNet": [], "Wav2Lip": [], "videos": []}

        for result in results:
            if "ground_truth" in result and "metrics" in result["lipnet"] and "metrics" in result["wav2lip"]:
                video_name = os.path.basename(result["video_path"])
                video_cers["videos"].append(video_name)
                video_cers["LipNet"].append(result["lipnet"]["metrics"]["CER"])
                video_cers["Wav2Lip"].append(result["wav2lip"]["metrics"]["CER"])

        if video_cers["videos"]:
            plt.figure(figsize=(12, 6))
            x = np.arange(len(video_cers["videos"]))
            width = 0.35

            plt.bar(x - width/2, video_cers["LipNet"], width, label='LipNet')
            plt.bar(x + width/2, video_cers["Wav2Lip"], width, label='Wav2Lip')

            plt.ylabel('Character Error Rate (CER)')
            plt.title('CER by Video and Model')
            plt.xticks(x, video_cers["videos"], rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()

            plt.savefig(os.path.join(output_dir, 'per_video_cer_comparison.png'))
            plt.close()

        print(f"\nComparison visualizations saved to {output_dir} directory.")

def main():
    parser = argparse.ArgumentParser(description='Lip Reading Model Comparison')
    parser.add_argument('--video', type=str, help='Path to a single video file to process')
    parser.add_argument('--video_dir', type=str, help='Path to directory containing multiple videos')
    parser.add_argument('--ground_truth', type=str, help='Path to ground truth text file (video_name,text format)')
    parser.add_argument('--output_dir', type=str, default='comparison_results', help='Directory to save results')

    args = parser.parse_args()

    lip_reading = LipReadingComparison()

    if args.video:
        # Process single video
        result = lip_reading.process_video(args.video)
        if result:
            summary = {"lipnet": {}, "wav2lip": {}}
            # if "metrics" in result["lipnet"] and "metrics" in result["wav2lip"]:
            if "metrics" in result["lipnet"]:
                summary["lipnet"] = {
                    "avg_CER": result["lipnet"]["metrics"]["CER"],
                    "avg_WER": result["lipnet"]["metrics"]["WER"],
                    "avg_processing_time": result["lipnet"]["processing_time"]
                }
                # summary["wav2lip"] = {
                #     "avg_CER": result["wav2lip"]["metrics"]["CER"],
                #     "avg_WER": result["wav2lip"]["metrics"]["WER"],
                #     "avg_processing_time": result["wav2lip"]["processing_time"]
                # }
            lip_reading.visualize_comparison([result], summary, args.output_dir)

    elif args.video_dir:
        # Process batch of videos
        results, summary = lip_reading.process_batch(args.video_dir, args.ground_truth)
        lip_reading.visualize_comparison(results, summary, args.output_dir)

    else:
        print("Please provide either --video or --video_dir argument")
        parser.print_help()

if __name__ == "__main__":
    main()