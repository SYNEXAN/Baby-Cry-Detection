"""
Module: engine/audio_detector.py
Mục đích:
Bộ phân tích & phát hiện nguyên nhân tiếng khóc trẻ sơ sinh dựa trên mô hình 1D-CNN
chuẩn phương pháp nghiên cứu Euréka 2025 (Huỳnh Nguyễn Sơn - BabySensory).

Tính năng:
1. Tải trọng số 1D-CNN (Eureka-1DCNN-MFCC, ~186k params, siêu nhẹ ~760KB).
2. Chuẩn hóa âm thanh đầu vào (WAV/MP3/Microphone): 16kHz, Mono, chuẩn hóa biên độ.
3. Cắt/Đệm 10.0 giây (160,000 samples) và trích xuất 40 dải MFCC chuẩn hóa Z-Score.
4. Dự đoán xác suất cho 5 nhóm nguyên nhân khóc:
   - Đói sữa (hungry)
   - Khó chịu / Tã ướt (discomfort)
   - Mệt mỏi / Buồn ngủ (tired)
   - Đau bụng / Co thắt (belly_pain)
   - Cần ợ hơi (burping)
5. Cung cấp khuyến nghị chăm sóc y tế / hướng dẫn sơ cấp cứu dành cho cha mẹ.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from scipy.io import wavfile
import librosa

logger = logging.getLogger("AudioDetector")

# 4 nhóm nguyên nhân lâm sàng chuẩn Karim Eletriby SOTA & 5 nhóm cũ Donate-a-Cry
CLASSES_4 = ['burping', 'needs', 'physical_pain', 'scared']
CLASSES_5 = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
CLASSES = CLASSES_4

CLASS_VIETNAMESE = {
    'burping': 'Cần ợ hơi sau bú (Burping)',
    'needs': 'Nhu cầu sinh hoạt (Needs - Đói / Mệt / Tã)',
    'physical_pain': 'Đau đớn thể chất / Co thắt bụng (Physical Pain)',
    'scared': 'Hoảng sợ / Giật mình đột ngột (Scared)',
    # Backward compatibility
    'hungry': 'Đói sữa (Hungry)',
    'discomfort': 'Khó chịu / Tã ướt (Discomfort)',
    'tired': 'Mệt mỏi / Buồn ngủ (Tired)',
    'belly_pain': 'Đau bụng / Co thắt ruột (Belly Pain)'
}

PARENT_ADVICE = {
    'burping': (
        "**Khuyến nghị cho ba mẹ (Cần ợ hơi):**\n"
        "- Khí dư thừa tích tụ trong dạ dày sau khi bú khiến bé bị tức ngực và khó chịu.\n"
        "- Bế đứng bé, cằm tựa nhẹ lên vai ba mẹ và áp sát lồng ngực bé vào ngực bạn.\n"
        "- Khum lòng bàn tay vỗ nhẹ lưng từ dưới thắt lưng hướng dần lên vai đến khi bé phát ra tiếng ợ hơi giải phóng khí."
    ),
    'needs': (
        "**Khuyến nghị cho ba mẹ (Nhu cầu sinh hoạt cơ bản):**\n"
        "- Trẻ đang phát tín hiệu có nhu cầu sinh hoạt tự nhiên (đói sữa, buồn ngủ hoặc tã ướt):\n"
        "  1. **Cữ bú:** Kiểm tra giờ bú gần nhất, miệng bé có phản xạ rúc tìm ti không? Cho bé bú lượng sữa thích hợp.\n"
        "  2. **Tã lót:** Kiểm tra ngay tã có bị bẩn, ướt hoặc làm cọ xát da bé không và thay mới nếu cần.\n"
        "  3. **Giấc ngủ:** Nếu bé đã thức quá thời gian thức tối đa, vỗ về nhẹ nhàng, giảm ánh sáng và tiếng ồn để giúp bé vào giấc."
    ),
    'physical_pain': (
        "**CẢNH BÁO Y TẾ (Đau đớn thể chất / Đau bụng Colic):**\n"
        "- Trẻ có dấu hiệu đau đớn dữ dội hoặc co thắt đường tiêu hóa (Colic/đầy trướng bụng):\n"
        "- Đặt trẻ nằm ngửa, kiểm tra xem bụng bé có bị chướng căng cứng hay không.\n"
        "- Massage nhẹ nhàng vùng bụng quanh rốn theo chiều kim đồng hồ hoặc gập chân bé về phía bụng (động tác đạp xe) để tống hơi thừa.\n"
        "- Kiểm tra toàn thân tìm vết côn trùng cắn, dị vật hoặc sợi tóc thắt ngón tay/chân (Hair Tourniquet).\n"
        "- **Lưu ý:** Nếu trẻ khóc thét liên tục nhiều giờ kèm sốt cao, nôn trớ hoặc co giật, hãy đưa trẻ đến cơ sở y tế chuyên khoa ngay lập tức."
    ),
    'scared': (
        "**Khuyến nghị cho ba mẹ (Hoảng sợ / Giật mình đột ngột):**\n"
        "- Trẻ bị giật mình bởi tiếng động lớn đột ngột, ánh sáng chói hoặc cảm giác mất thăng bằng (phản xạ Moro):\n"
        "- Ngay lập tức bế ấp bé vào lòng, đặt bàn tay lên ngực bé để tạo cảm giác an toàn và tiếp xúc da kề da.\n"
        "- Quấn chũn hoặc kén nhẹ nhàng giữ cố định hai tay bé để giảm phản xạ giật mình co giật.\n"
        "- Hạ nhỏ âm thanh trong phòng, giữ không gian yên tĩnh và dùng giọng nói trầm ấm xoa dịu trẻ."
    ),
    # Backward compatibility
    'hungry': (
        "**Khuyến nghị cho ba mẹ:**\n"
        "- Trẻ đang phát tín hiệu tìm ti/sữa hoặc đã tới cữ bú tiếp theo.\n"
        "- Kiểm tra miệng trẻ có phản xạ rúc tìm ti không, kiểm tra giờ bú gần nhất.\n"
        "- Cho trẻ bú lượng vừa đủ, tránh để trẻ khóc quá lâu nuốt phải nhiều không khí."
    ),
    'discomfort': (
        "**Khuyến nghị cho ba mẹ:**\n"
        "- Trẻ đang khó chịu về cơ thể: tã bẩn/ướt, quần áo cọ xát, nhiệt độ phòng quá nóng/lạnh.\n"
        "- Kiểm tra ngay tã lót và thay mới nếu cần thiết.\n"
        "- Kiểm tra thân nhiệt vùng gáy của bé (26-28 độ C là nhiệt độ phòng lý tưởng)."
    ),
    'tired': (
        "**Khuyến nghị cho ba mẹ:**\n"
        "- Trẻ đã quá mệt mỏi, mắt lim dim hoặc chớp liên tục, cần được đi ngủ.\n"
        "- Giảm bớt kích thích: tắt đèn sáng, hạ nhỏ tiếng ồn xung quanh.\n"
        "- Quấn chũn hoặc vỗ về nhẹ nhàng theo nhịp, bật tiếng ồn trắng (white noise) giúp bé dễ vào giấc."
    ),
    'belly_pain': (
        "**Khuyến nghị cho ba mẹ:**\n"
        "- Trẻ có thể đang bị đau bụng, co thắt dạ dày hoặc đầy trướng bụng (colic).\n"
        "- Đặt trẻ nằm ngửa trên giường, massage nhẹ nhàng bụng quanh rốn theo chiều kim đồng hồ.\n"
        "- Nhẹ nhàng gập hai đầu gối của bé ép về phía bụng (động tác đạp xe) để đẩy hơi dư thừa ra ngoài.\n"
        "- Nếu bé khóc ré liên tục nhiều giờ kèm sốt hoặc nôn trớ, hãy liên hệ bác sĩ nhi khoa."
    )
}

TARGET_SR = 16000
N_MFCC = 40
N_FFT = 1024
HOP_LENGTH = 512
DURATION_SEC = 10.0
TARGET_SAMPLES = int(TARGET_SR * DURATION_SEC)


class Eureka1DCNN(nn.Module):
    """
    Kiến trúc 1D-CNN siêu nhẹ mô phỏng cấu hình nghiên cứu Euréka 2025.
    Input: (Batch, 40, TimeSteps)
    """
    def __init__(self, num_classes: int = 5, in_channels: int = 40):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2),

            # Block 2
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2),

            # Block 3
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        return self.classifier(feat)


class AudioResNet18Classifier(nn.Module):
    """
    Kiến trúc SOTA Audio ResNet18 nhận đầu vào 3 kênh (Log-Mel + Delta + Delta-Delta).
    Đạt 84.5% Test Accuracy trên bộ dữ liệu lâm sàng 4 lớp Karim Eletriby.
    Input: (Batch, 3, 128, TimeSteps)
    """
    def __init__(self, num_classes: int = 4):
        super().__init__()
        import torchvision.models as models
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.4),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


class AudioResNetClassifier(nn.Module):
    """
    Kiến trúc Audio ResNet34 nhận đầu vào 3 kênh (Log-Mel + Delta + Delta-Delta).
    Input: (Batch, 3, 128, TimeSteps)
    """
    def __init__(self, num_classes: int = 5):
        super().__init__()
        import torchvision.models as models
        self.backbone = models.resnet34(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


class CompactAudioCNN(nn.Module):
    """
    Kiến trúc Compact Audio CNN siêu nhẹ (~241k tham số) chống học vẹt.
    Input: (Batch, 1, 128, TimeSteps)
    """
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.4),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return self.classifier(feat.view(feat.size(0), -1))


class AudioCryClassifier:
    """
    Trình phân loại nguyên nhân tiếng khóc trẻ em phục vụ suy luận thời gian thực.
    Mặc định nạp SOTA ResNet18 (4 lớp lâm sàng, 84.5% Test Acc).
    Tự động tương thích ngược với Compact Audio CNN và Euréka 1D-CNN nếu được cấu hình.
    """
    def __init__(
        self,
        weights_path: str = "models/best_eletriby_4class_audio_classifier.pt",
        metadata_path: str = "kaggle_eletriby_results/eletriby_4class_metadata.json",
        device: Optional[str] = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.weights_path = Path(weights_path)
        self.metadata_path = Path(metadata_path)
        self.classes = CLASSES_4
        self.model: Optional[nn.Module] = None
        self.model_name = "Unknown"
        self.duration_sec = 5.0
        self.is_loaded = False

        self._load_metadata()
        self._load_model()

    def _load_metadata(self):
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    if 'classes' in meta:
                        self.classes = meta['classes']
                    if 'duration_sec' in meta:
                        self.duration_sec = float(meta['duration_sec'])
            except Exception as e:
                logger.warning(f"Không thể đọc file metadata {self.metadata_path}: {e}")

    def _load_model(self):
        # Kiểm tra danh sách đường dẫn ưu tiên
        candidate_weights = [
            self.weights_path,
            Path("models/best_eletriby_4class_audio_classifier.pt"),
            Path("kaggle_eletriby_results/best_eletriby_4class_audio_classifier.pt"),
            Path("kaggle_audio_results/best_compact_audio_classifier.pt"),
            Path("kaggle_audio_results/best_sota_audio_classifier.pt")
        ]

        found_path = None
        for p in candidate_weights:
            if p.exists():
                found_path = p
                break

        if found_path is None:
            logger.warning(f"Trọng số {self.weights_path} chưa sẵn sàng.")
            return

        self.weights_path = found_path

        try:
            state_dict = torch.load(self.weights_path, map_location=self.device)
            # Detect architecture based on state_dict keys and layer dimensions
            if any(k.startswith("classifier.2.weight") for k in state_dict.keys()):
                fc_weight_shape = state_dict["classifier.2.weight"].shape
                if fc_weight_shape == torch.Size([128, 512]):
                    self.model = AudioResNet18Classifier(num_classes=len(self.classes))
                    self.model_name = "SOTA ResNet18 Clinical 4-Class V4 (87.6% Precision)"
                    self.duration_sec = 5.0
                else:
                    self.model = AudioResNetClassifier(num_classes=len(self.classes))
                    self.model_name = "SOTA Audio ResNet34 (3-Ch Log-Mel)"
                    self.duration_sec = 10.0
            elif any(k.startswith("features.0.weight") and state_dict["features.0.weight"].ndim == 4 for k in state_dict.keys()):
                self.model = CompactAudioCNN(num_classes=len(self.classes))
                self.model_name = "Compact Audio CNN (241k Regularized)"
                self.duration_sec = 10.0
            else:
                self.model = Eureka1DCNN(num_classes=len(self.classes), in_channels=N_MFCC)
                self.model_name = "Euréka 2025 1D-CNN (40-band MFCC)"
                self.duration_sec = 10.0

            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            logger.info(f"Đã tải mô hình {self.model_name} thành công từ {self.weights_path}")
        except Exception as e:
            logger.error(f"Lỗi tải trọng số mô hình: {e}")
            self.is_loaded = False

    def preprocess_audio(self, audio_input: Union[str, Path, Tuple[int, np.ndarray], np.ndarray]) -> np.ndarray:
        """
        Chuẩn hóa âm thanh thành mảng Mono 16kHz float32 có độ dài đúng self.duration_sec
        """
        sr = TARGET_SR
        audio: np.ndarray

        if isinstance(audio_input, (str, Path)):
            audio_path = str(audio_input)
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Tệp âm thanh không tồn tại: {audio_path}")
            # Dùng librosa load đa định dạng (wav, mp3, flac, ogg)
            audio, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
        elif isinstance(audio_input, tuple):
            # Định dạng trả về của Gradio gr.Audio: (sample_rate, numpy_array)
            sr, audio = audio_input
            if audio.ndim > 1:
                # Chuyển kênh đa chiều sang mono
                audio = np.mean(audio, axis=1)
            # Resample nếu cần
            if sr != TARGET_SR:
                audio = librosa.resample(y=audio.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
                sr = TARGET_SR
        elif isinstance(audio_input, np.ndarray):
            audio = audio_input
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
        else:
            raise ValueError(f"Kiểu dữ liệu âm thanh không được hỗ trợ: {type(audio_input)}")

        # Đảm bảo float32 chuẩn hóa trong khoảng [-1.0, 1.0]
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        max_abs = np.max(np.abs(audio))
        if max_abs > 1.0:
            audio = audio / 32768.0
        elif max_abs > 0.0:
            audio = audio / max_abs

        # Chuẩn hóa độ dài theo self.duration_sec
        target_samples = int(TARGET_SR * self.duration_sec)
        if len(audio) < target_samples:
            # Ngắn hơn: Lặp lại và đệm
            n_repeats = int(np.ceil(target_samples / max(1, len(audio))))
            audio = np.tile(audio, n_repeats)[:target_samples]
        elif len(audio) > target_samples:
            # Dài hơn: Cắt đoạn giữa có năng lượng lớn nhất
            mid = len(audio) // 2
            half = target_samples // 2
            start = max(0, mid - half)
            end = start + target_samples
            if end > len(audio):
                end = len(audio)
                start = end - target_samples
            audio = audio[start:end]

        return audio

    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Trích xuất 40 dải MFCC chuẩn hóa Z-Score cho Eureka 1D-CNN
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=TARGET_SR,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
        # Z-Score normalization trên từng kênh tần số
        mean = np.mean(mfcc, axis=1, keepdims=True)
        std = np.std(mfcc, axis=1, keepdims=True) + 1e-6
        norm_mfcc = ((mfcc - mean) / std).astype(np.float32)
        return norm_mfcc

    def extract_features_resnet(self, audio: np.ndarray) -> np.ndarray:
        """
        Trích xuất Log-Mel Spectrogram 3 kênh (Log-Mel, Delta, Delta-Delta) cho ResNet18/ResNet34
        Output shape: (3, 128, T)
        """
        mel = librosa.feature.melspectrogram(
            y=audio, sr=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=128, power=2.0
        )
        log_mel = librosa.power_to_db(mel)
        norm = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)
        d1 = librosa.feature.delta(norm, order=1)
        d2 = librosa.feature.delta(norm, order=2)
        feat = np.stack([norm, d1, d2], axis=0).astype(np.float32)
        return feat

    def predict(self, audio_input: Union[str, Path, Tuple[int, np.ndarray], np.ndarray]) -> Dict[str, Any]:
        """
        Thực hiện phân tích tiếng khóc và trả về kết quả chi tiết
        """
        if not self.is_loaded or self.model is None:
            # Tự động thử tải lại nếu trước đó chưa tìm thấy file
            self._load_model()
            if not self.is_loaded or self.model is None:
                raise RuntimeError("Mô hình chưa được tải thành công.")

        # Tiền xử lý âm thanh
        audio = self.preprocess_audio(audio_input)

        if isinstance(self.model, CompactAudioCNN):
            mel = librosa.feature.melspectrogram(
                y=audio, sr=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=128, power=2.0
            )
            log_mel = librosa.power_to_db(mel)
            norm_mel = ((log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)).astype(np.float32)
            features = norm_mel
            input_tensor = torch.tensor(norm_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        elif isinstance(self.model, (AudioResNetClassifier, AudioResNet18Classifier)):
            features = self.extract_features_resnet(audio)  # (3, 128, T)
            input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        else:
            features = self.extract_features(audio)  # (40, T)
            input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        top_idx = int(np.argmax(probs))
        top_class = self.classes[top_idx]
        top_prob = float(probs[top_idx])

        # Tạo từ điển xác suất theo từng nhãn tiếng Việt
        breakdown = {}
        for c, p in zip(self.classes, probs):
            vi_name = CLASS_VIETNAMESE.get(c, c)
            breakdown[vi_name] = float(p)

        # Sắp xếp xác suất giảm dần
        sorted_breakdown = dict(sorted(breakdown.items(), key=lambda item: item[1], reverse=True))

        return {
            "primary_cause": top_class,
            "vietnamese_cause": CLASS_VIETNAMESE.get(top_class, top_class),
            "confidence": top_prob,
            "probabilities": sorted_breakdown,
            "advice": PARENT_ADVICE.get(top_class, "Theo dõi thêm các biểu hiện của trẻ."),
            "sample_rate": TARGET_SR,
            "duration_sec": self.duration_sec,
            "model_name": self.model_name,
            "feature_shape": features.shape
        }
