#!/usr/bin/env python3
"""
Multimodal Data Preparation for Qwen3-Omni
Handles text, audio, image, and video data preprocessing
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random
from dataclasses import dataclass

import numpy as np
import torch
import torchaudio
import torchvision
from torchvision import transforms
from PIL import Image
import cv2
import soundfile as sf
import librosa
from tqdm import tqdm
import webdataset as wds


@dataclass
class MultimodalSample:
    """Container for multimodal data sample"""
    conversation_id: str
    messages: List[Dict]
    audio_paths: Optional[List[str]] = None
    image_paths: Optional[List[str]] = None
    video_paths: Optional[List[str]] = None
    metadata: Optional[Dict] = None


class AudioProcessor:
    """Process audio data for Qwen3-Omni"""

    def __init__(self, sample_rate: int = 24000, max_length: int = 300):
        self.sample_rate = sample_rate
        self.max_length = max_length  # seconds

    def process_audio(self, audio_path: str) -> Dict:
        """Load and preprocess audio file"""
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)

        # Trim or pad to max length
        max_samples = self.sample_rate * self.max_length
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        elif len(audio) < max_samples:
            audio = np.pad(audio, (0, max_samples - len(audio)))

        # Extract features
        features = {
            "waveform": audio,
            "sample_rate": self.sample_rate,
            "duration": len(audio) / self.sample_rate,
            "rms_energy": librosa.feature.rms(y=audio)[0],
            "zero_crossing_rate": librosa.feature.zero_crossing_rate(audio)[0]
        }

        # Optional: Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate, n_mels=128
        )
        features["mel_spectrogram"] = librosa.power_to_db(mel_spec)

        return features

    def augment_audio(self, audio: np.ndarray, augmentation_prob: float = 0.5) -> np.ndarray:
        """Apply audio augmentation"""
        if random.random() < augmentation_prob:
            # Add noise
            if random.random() < 0.3:
                noise = np.random.normal(0, 0.01, audio.shape)
                audio = audio + noise

            # Time stretch
            if random.random() < 0.3:
                rate = random.uniform(0.9, 1.1)
                audio = librosa.effects.time_stretch(audio, rate=rate)

            # Pitch shift
            if random.random() < 0.3:
                n_steps = random.randint(-2, 2)
                audio = librosa.effects.pitch_shift(
                    audio, sr=self.sample_rate, n_steps=n_steps
                )

        return audio


class VideoProcessor:
    """Process video data for Qwen3-Omni"""

    def __init__(self, fps: int = 2, max_frames: int = 240, size: Tuple[int, int] = (224, 224)):
        self.fps = fps
        self.max_frames = max_frames
        self.size = size

    def process_video(self, video_path: str, extract_audio: bool = True) -> Dict:
        """Extract frames and audio from video"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame sampling interval
        sample_interval = max(1, int(fps / self.fps))
        frames = []

        frame_count = 0
        while cap.isOpened() and len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_interval == 0:
                # Resize frame
                frame = cv2.resize(frame, self.size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            frame_count += 1

        cap.release()

        features = {
            "frames": np.array(frames),
            "num_frames": len(frames),
            "original_fps": fps,
            "sampled_fps": self.fps,
            "duration": total_frames / fps
        }

        # Extract audio if requested
        if extract_audio:
            audio_features = self.extract_audio_from_video(video_path)
            features["audio"] = audio_features

        return features

    def extract_audio_from_video(self, video_path: str) -> Optional[Dict]:
        """Extract audio track from video"""
        try:
            import subprocess
            import tempfile

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                cmd = [
                    'ffmpeg', '-i', video_path,
                    '-vn', '-acodec', 'pcm_s16le',
                    '-ar', '24000', '-ac', '1',
                    tmp.name, '-y'
                ]
                subprocess.run(cmd, capture_output=True, check=True)

                audio, sr = sf.read(tmp.name)
                os.unlink(tmp.name)

                return {
                    "waveform": audio,
                    "sample_rate": sr
                }
        except Exception as e:
            print(f"Failed to extract audio from video: {e}")
            return None


class ImageProcessor:
    """Process image data for Qwen3-Omni"""

    def __init__(self, size: Tuple[int, int] = (336, 336)):
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def process_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image"""
        image = Image.open(image_path).convert('RGB')
        return self.transform(image)

    def augment_image(self, image: Image.Image, augmentation_prob: float = 0.5) -> Image.Image:
        """Apply image augmentation"""
        if random.random() < augmentation_prob:
            augment = transforms.Compose([
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ])
            image = augment(image)
        return image


class DatasetBuilder:
    """Build multimodal dataset for Qwen3-Omni"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.audio_processor = AudioProcessor()
        self.video_processor = VideoProcessor()
        self.image_processor = ImageProcessor()

    def create_conversation(self, sample: MultimodalSample) -> Dict:
        """Create conversation format for training"""
        conversation = {
            "conversation_id": sample.conversation_id,
            "conversations": sample.messages
        }

        # Add multimodal paths
        if sample.audio_paths:
            conversation["audio_paths"] = sample.audio_paths
        if sample.image_paths:
            conversation["image_paths"] = sample.image_paths
        if sample.video_paths:
            conversation["video_paths"] = sample.video_paths

        # Add metadata
        if sample.metadata:
            conversation["metadata"] = sample.metadata

        return conversation

    def process_dataset(self, samples: List[MultimodalSample],
                       split: str = "train") -> None:
        """Process and save dataset"""
        output_file = self.output_dir / f"{split}.jsonl"

        with open(output_file, 'w') as f:
            for sample in tqdm(samples, desc=f"Processing {split} data"):
                conversation = self.create_conversation(sample)
                f.write(json.dumps(conversation) + '\n')

        print(f"Saved {len(samples)} samples to {output_file}")

    def create_webdataset(self, samples: List[MultimodalSample],
                         split: str = "train",
                         shard_size: int = 1000) -> None:
        """Create WebDataset format for efficient loading"""
        output_pattern = str(self.output_dir / f"{split}-%06d.tar")

        with wds.ShardWriter(output_pattern, maxcount=shard_size) as sink:
            for idx, sample in enumerate(tqdm(samples, desc=f"Creating WebDataset {split}")):
                key = f"{split}_{idx:08d}"

                # Prepare sample dict
                sample_dict = {
                    "__key__": key,
                    "json": json.dumps(self.create_conversation(sample))
                }

                # Add multimodal data
                if sample.audio_paths:
                    for i, audio_path in enumerate(sample.audio_paths):
                        with open(audio_path, 'rb') as f:
                            sample_dict[f"audio_{i}.wav"] = f.read()

                if sample.image_paths:
                    for i, image_path in enumerate(sample.image_paths):
                        with open(image_path, 'rb') as f:
                            sample_dict[f"image_{i}.jpg"] = f.read()

                if sample.video_paths:
                    for i, video_path in enumerate(sample.video_paths):
                        with open(video_path, 'rb') as f:
                            sample_dict[f"video_{i}.mp4"] = f.read()

                sink.write(sample_dict)

        print(f"Created WebDataset shards in {self.output_dir}")


def create_sample_data():
    """Create sample multimodal conversations"""
    samples = []

    # Example 1: Image + Text
    sample1 = MultimodalSample(
        conversation_id="sample_001",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "data/images/cat.jpg"},
                    {"type": "text", "text": "What animal is in this image?"}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "This is a cat. It appears to be a domestic shorthair cat with tabby markings."}
                ]
            }
        ],
        image_paths=["data/images/cat.jpg"]
    )
    samples.append(sample1)

    # Example 2: Audio + Text
    sample2 = MultimodalSample(
        conversation_id="sample_002",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": "data/audio/speech.wav"},
                    {"type": "text", "text": "Transcribe this audio and identify the language."}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "The audio contains English speech saying: 'Hello, how are you today?'"}
                ]
            }
        ],
        audio_paths=["data/audio/speech.wav"]
    )
    samples.append(sample2)

    # Example 3: Video + Audio + Text
    sample3 = MultimodalSample(
        conversation_id="sample_003",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "data/videos/tutorial.mp4"},
                    {"type": "text", "text": "Summarize what's happening in this video."}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "This is a tutorial video showing how to..."}
                ]
            }
        ],
        video_paths=["data/videos/tutorial.mp4"],
        metadata={"duration": 60, "has_audio": True}
    )
    samples.append(sample3)

    return samples


def main():
    parser = argparse.ArgumentParser(description="Prepare multimodal data for Qwen3-Omni")
    parser.add_argument("--input_dir", help="Input directory with raw data")
    parser.add_argument("--output_dir", default="data/processed", help="Output directory")
    parser.add_argument("--format", choices=["jsonl", "webdataset"], default="jsonl")
    parser.add_argument("--create_sample", action="store_true", help="Create sample data")

    args = parser.parse_args()

    builder = DatasetBuilder(args.output_dir)

    if args.create_sample:
        # Create sample data
        samples = create_sample_data()
        builder.process_dataset(samples, split="sample")
        print("Created sample dataset")
    else:
        # Process actual data
        # This would load your real dataset
        pass


if __name__ == "__main__":
    main()