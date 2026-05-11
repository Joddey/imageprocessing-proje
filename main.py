import sys
import os
import urllib.request
import threading
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QCheckBox, QGridLayout,
                             QFileDialog)
from PyQt5.QtCore import Qt, QMimeData, QPoint, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QDrag, QPixmap, QImage
from PyQt5.QtWebEngineWidgets import QWebEngineView

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
    MEDIAPIPE_AVAILABLE = True
except Exception as e:
    print(f"MediaPipe yüklenemedi: {e}")
    MEDIAPIPE_AVAILABLE = False

try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except Exception as e:
    print(f"SpeechRecognition yüklenemedi: {e}")
    SPEECH_AVAILABLE = False

try:
    import vosk
    import sounddevice as sd
    import queue, json
    VOSK_AVAILABLE = True
except Exception as e:
    print(f"Vosk yüklenemedi: {e}")
    VOSK_AVAILABLE = False

try:
    from PIL import Image as PILImage
    import torch
    from transformers import AutoImageProcessor, SiglipForImageClassification
    import easyocr
    PHOTO_AVAILABLE = True
except Exception as e:
    print(f"Fotoğraf analizi yüklenemedi: {e}")
    PHOTO_AVAILABLE = False

VOSK_MODEL_PATH = "vosk-model-small-tr-0.3"

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# Sesli komut eşleştirme tablosu (Türkçe)
VOICE_WEATHER_MAP = {
    'güneşli': '☀️ Güneşli',
    'güneş': '☀️ Güneşli',
    'sunny': '☀️ Güneşli',
    'yağmurlu': '🌧️ Yağmurlu',
    'yağmur': '🌧️ Yağmurlu',
    'rain': '🌧️ Yağmurlu',
    'karlı': '❄️ Karlı',
    'snow': '❄️ Karlı',
    'parçalı': '⛅ Parçalı Bulutlu',
    'bulutlu': '⛅ Parçalı Bulutlu',
    'bulut': '⛅ Parçalı Bulutlu',
    'cloud': '⛅ Parçalı Bulutlu',
    'fırtınalı': '🌩️ Fırtınalı',
    'fırtına': '🌩️ Fırtınalı',
    'storm': '🌩️ Fırtınalı',
    'sisli': '🌫️ Sisli',
    'sis': '🌫️ Sisli',
    'fog': '🌫️ Sisli',
}

# Türkiye illeri (GeoJSON'daki isimlerle eşleşmeli)
TR_CITIES = [
    'adana', 'adıyaman', 'afyonkarahisar', 'afyon', 'ağrı', 'amasya', 'ankara', 'antalya', 'artvin',
    'aydın', 'balıkesir', 'bilecik', 'bingöl', 'bitlis', 'bolu', 'burdur', 'bursa',
    'çanakkale', 'çankırı', 'çorum', 'denizli', 'diyarbakır', 'edirne', 'elazığ', 'erzincan',
    'erzurum', 'eskişehir', 'gaziantep', 'giresun', 'gümüşhane', 'hakkari', 'hatay', 'isparta',
    'mersin', 'istanbul', 'izmir', 'kars', 'kastamonu', 'kayseri', 'kırklareli', 'kırşehir',
    'kocaeli', 'konya', 'kütahya', 'malatya', 'manisa', 'kahramanmaraş', 'mardin', 'muğla',
    'muş', 'nevşehir', 'niğde', 'ordu', 'rize', 'sakarya', 'samsun', 'siirt', 'sinop',
    'sivas', 'tekirdağ', 'tokat', 'trabzon', 'tunceli', 'şanlıurfa', 'uşak', 'van',
    'yozgat', 'zonguldak', 'aksaray', 'bayburt', 'karaman', 'kırıkkale', 'batman', 'şırnak',
    'bartın', 'ardahan', 'iğdır', 'yalova', 'karabük', 'kilis', 'osmaniye', 'düzce',
]


VOICE_REGION_MAP = {
    'marmara': 'Marmara',
    'ege': 'Ege',
    'akdeniz': 'Akdeniz',
    'iç anadolu': 'İç Anadolu',
    'karadeniz': 'Karadeniz',
    'doğu anadolu': 'Doğu Anadolu',
    'güneydoğu': 'Güneydoğu',
    'güneydoğu anadolu': 'Güneydoğu',
}


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📥 Hand Landmarker modeli indiriliyor...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("✅ Model indirildi!")
            return True
        except Exception as e:
            print(f"❌ Model indirilemedi: {e}")
            return False
    return True


class VoiceWorker(QObject):
    """Vosk ile offline ses tanıma"""
    weather_detected = pyqtSignal(str, str)  # weather_type, city
    region_detected = pyqtSignal(str)         # region name
    status_changed = pyqtSignal(str)

    SAMPLE_RATE = 16000

    def __init__(self):
        super().__init__()
        self.running = False
        self.q = queue.Queue()

    def start_listening(self):
        self.running = True
        thread = threading.Thread(target=self._listen_loop, daemon=True)
        thread.start()

    def stop_listening(self):
        self.running = False

    def _listen_loop(self):
        if not VOSK_AVAILABLE:
            self.status_changed.emit("🎤 Vosk yüklenemedi")
            return

        if not os.path.exists(VOSK_MODEL_PATH):
            self.status_changed.emit(f"🎤 Model bulunamadı: {VOSK_MODEL_PATH}")
            return

        try:
            vosk.SetLogLevel(-1)
            model = vosk.Model(VOSK_MODEL_PATH)
            rec = vosk.KaldiRecognizer(model, self.SAMPLE_RATE)
        except Exception as e:
            self.status_changed.emit(f"🎤 Model hatası: {e}")
            return

        def callback(indata, frames, time, status):
            self.q.put(bytes(indata))

        self.status_changed.emit("🎤 Dinleniyor...")
        try:
            with sd.RawInputStream(samplerate=self.SAMPLE_RATE, blocksize=8000,
                                   dtype='int16', channels=1, callback=callback):
                while self.running:
                    data = self.q.get()
                    if rec.AcceptWaveform(data):
                        result = json.loads(rec.Result())
                        text = result.get('text', '').lower().strip()
                        if text:
                            print(f"🎙️ Duyulan: {text}")
                            region = self._match_region(text)
                            weather = self._match_weather(text)
                            city = self._match_city(text)
                            if region:
                                self.region_detected.emit(region)
                                self.status_changed.emit(f"📍 {region}")
                            elif weather:
                                self.weather_detected.emit(weather, city or '')
                                msg = f"✅ {weather}"
                                if city:
                                    msg += f" → {city}"
                                self.status_changed.emit(msg)
                            elif city:
                                self.status_changed.emit(f"🏙️ {city.title()} (hava durumu?)")
                            else:
                                self.status_changed.emit(f"❓ {text}")
                    else:
                        partial = json.loads(rec.PartialResult())
                        p = partial.get('partial', '').strip()
                        if p:
                            self.status_changed.emit(f"🔄 {p}...")
        except Exception as e:
            self.status_changed.emit(f"🎤 Hata: {e}")
            print(f"Ses hatası: {e}")

    def _match_weather(self, text):
        for keyword, weather_type in VOICE_WEATHER_MAP.items():
            if keyword in text:
                return weather_type
        return None

    def _match_city(self, text):
        for city in TR_CITIES:
            if city in text:
                return city
        return None

    def _match_region(self, text):
        # Normalize - Vosk bazen i̇ (noktalı) yazıyor
        text = text.replace('i̇', 'i')
        if 'karadeniz' in text or 'kara deniz' in text:
            return 'Karadeniz'
        if 'ic anadolu' in text or 'iç anadolu' in text or ('ic' in text and 'anadolu' in text) or ('iç' in text and 'anadolu' in text):
            return 'İç Anadolu'
        if 'dogu anadolu' in text or 'doğu anadolu' in text or ('dogu' in text and 'anadolu' in text) or ('doğu' in text and 'anadolu' in text):
            return 'Doğu Anadolu'
        if 'guneydogu' in text or 'güneydoğu' in text or 'guneydoğu' in text:
            return 'Güneydoğu'
        if 'marmara' in text:
            return 'Marmara'
        if 'akdeniz' in text:
            return 'Akdeniz'
        if 'ege' in text:
            return 'Ege'
        return None


class PhotoAnalyzer(QObject):
    analysis_done = pyqtSignal(str, str)  # city, weather_type
    status_changed = pyqtSignal(str)

    # Model çıktısı → bizim hava durumu ikonlarımız
    LABEL_MAP = {
        'sun/clear':        '☀️ Güneşli',
        'cloudy/overcast':  '⛅ Parçalı Bulutlu',
        'rain/storm':       '🌧️ Yağmurlu',
        'snow/frosty':      '❄️ Karlı',
        'foggy/hazy':       '🌫️ Sisli',
    }

    def __init__(self):
        super().__init__()
        self.model = None
        self.processor = None
        self.ocr = None
        self._loaded = False

    def load_models(self):
        if self._loaded or not PHOTO_AVAILABLE:
            return
        try:
            self.status_changed.emit("🔄 Modeller yükleniyor...")
            self.processor = AutoImageProcessor.from_pretrained(
                'prithivMLmods/Weather-Image-Classification')
            self.model = SiglipForImageClassification.from_pretrained(
                'prithivMLmods/Weather-Image-Classification')
            self.model.eval()
            self.ocr = easyocr.Reader(['tr', 'en'], gpu=False)
            self._loaded = True
            self.status_changed.emit("✅ Modeller hazır")
        except Exception as e:
            self.status_changed.emit(f"❌ Model yüklenemedi: {e}")
            print(f"Model yüklenemedi: {e}")

    def analyze(self, image_path):
        if not PHOTO_AVAILABLE:
            return
        thread = threading.Thread(
            target=self._analyze_thread, args=(image_path,), daemon=True)
        thread.start()

    def _analyze_thread(self, image_path):
        try:
            print(f"📸 Analiz başladı: {image_path}")
            if not self._loaded:
                self.load_models()

            if not self._loaded:
                self.status_changed.emit("❌ Model yüklenemedi")
                return

            self.status_changed.emit("🔍 Fotoğraf analiz ediliyor...")
            img = PILImage.open(image_path).convert('RGB')
            img_cv = cv2.imread(image_path)
            print("✅ Fotoğraf yüklendi")

            # --- OCR: kırmızı yazıyı bul ---
            city = self._extract_red_text(img_cv)
            print(f"📍 OCR sonucu: {city}")
            self.status_changed.emit(f"📍 Bulunan il: {city or '?'}")

            # --- Hava durumu sınıflandırma ---
            print("🔄 Hava durumu sınıflandırılıyor...")
            inputs = self.processor(images=img, return_tensors='pt')
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(
                    outputs.logits, dim=1).squeeze().tolist()

            id2label = self.model.config.id2label
            best_idx = int(torch.argmax(outputs.logits))
            best_label = id2label[best_idx]
            weather = self.LABEL_MAP.get(best_label, '⛅ Parçalı Bulutlu')

            print(f"📸 Sonuç → OCR: {city} | Hava: {best_label} → {weather}")
            self.status_changed.emit(f"✅ {city or '?'} → {weather}")
            self.analysis_done.emit(city or '', weather)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status_changed.emit(f"❌ Hata: {e}")
            print(f"Analiz hatası: {e}")

    def _extract_red_text(self, img_cv):
        """Kırmızı renkli metni OCR ile çıkar"""
        try:
            hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)

            # Kırmızı renk maskesi (iki aralık)
            mask1 = cv2.inRange(hsv, np.array([0, 100, 100]),
                                np.array([10, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([160, 100, 100]),
                                np.array([180, 255, 255]))
            red_mask = cv2.bitwise_or(mask1, mask2)

            # Kırmızı bölgeyi beyaz metin siyah arka plan yap
            result = cv2.bitwise_and(img_cv, img_cv, mask=red_mask)
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

            # Kırmızı piksel yoksa tüm görüntüye OCR uygula
            if cv2.countNonZero(thresh) < 100:
                thresh = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

            results = self.ocr.readtext(thresh, detail=0)
            text = ' '.join(results).lower().strip()
            print(f"OCR ham: {text}")

            # İl adı eşleştir
            for city in TR_CITIES:
                if city in text:
                    return city
            return None
        except Exception as e:
            print(f"OCR hatası: {e}")
            return None


class CameraWidget(QLabel):
    hand_position_changed = pyqtSignal(float, float, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setStyleSheet("""
            QLabel {
                border: 3px solid #2C3E50;
                border-radius: 10px;
                background-color: black;
            }
        """)

        self.mediapipe_ok = False
        self.landmarker = None
        self.frame_timestamp = 0
        self.is_pinching = False

        if MEDIAPIPE_AVAILABLE:
            try:
                if download_model():
                    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
                    options = HandLandmarkerOptions(
                        base_options=base_options,
                        running_mode=RunningMode.VIDEO,
                        num_hands=1,
                        min_hand_detection_confidence=0.5,
                        min_hand_presence_confidence=0.5,
                        min_tracking_confidence=0.5
                    )
                    self.landmarker = HandLandmarker.create_from_options(options)
                    self.mediapipe_ok = True
                    print("✅ MediaPipe Hand Landmarker başlatıldı!")
                else:
                    print("❌ Model indirilemedi.")
            except Exception as e:
                print(f"❌ MediaPipe başlatılamadı: {e}")

        self.cap = None
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("❌ Kamera açılamadı!")
                self.show_error_message("Kamera bulunamadı")
            else:
                print("✅ Kamera başarıyla açıldı!")
        except Exception as e:
            print(f"❌ Kamera hatası: {e}")
            self.show_error_message(f"Kamera hatası: {e}")

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def show_error_message(self, message):
        self.setText(f"⚠️\n{message}")
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 3px solid #E74C3C; border-radius: 10px;
                background-color: #2C3E50; color: white; font-size: 14px;
            }
        """)

    def update_frame(self):
        if not self.cap or not self.cap.isOpened():
            return
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb_frame.shape

        if self.mediapipe_ok and self.landmarker:
            try:
                self.frame_timestamp += 33
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = self.landmarker.detect_for_video(mp_image, self.frame_timestamp)

                if result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]

                    connections = [
                        (0,1),(1,2),(2,3),(3,4),
                        (0,5),(5,6),(6,7),(7,8),
                        (0,9),(9,10),(10,11),(11,12),
                        (0,13),(13,14),(14,15),(15,16),
                        (0,17),(17,18),(18,19),(19,20),
                        (5,9),(9,13),(13,17)
                    ]
                    for a, b in connections:
                        ax, ay = int(landmarks[a].x * w), int(landmarks[a].y * h)
                        bx, by = int(landmarks[b].x * w), int(landmarks[b].y * h)
                        cv2.line(rgb_frame, (ax, ay), (bx, by), (0, 200, 200), 2)
                    for lm in landmarks:
                        cv2.circle(rgb_frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 255), -1)

                    thumb = landmarks[4]
                    index = landmarks[8]
                    thumb_px = (int(thumb.x * w), int(thumb.y * h))
                    index_px = (int(index.x * w), int(index.y * h))

                    dist = np.sqrt((thumb_px[0]-index_px[0])**2 + (thumb_px[1]-index_px[1])**2)
                    self.is_pinching = dist < 40

                    if self.is_pinching:
                        cv2.circle(rgb_frame, index_px, 18, (0, 255, 0), -1)
                        cv2.line(rgb_frame, thumb_px, index_px, (0, 255, 0), 3)
                        cv2.putText(rgb_frame, "TUTTU!", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    else:
                        cv2.circle(rgb_frame, index_px, 12, (255, 80, 80), -1)
                        cv2.circle(rgb_frame, index_px, 12, (255, 255, 255), 2)

                    self.hand_position_changed.emit(float(index.x), float(index.y), bool(self.is_pinching))
                else:
                    self.is_pinching = False

            except Exception as e:
                print(f"El algılama hatası: {e}")
        else:
            cv2.putText(rgb_frame, "El modu devre disi", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            cv2.putText(rgb_frame, "Fare ile kullanin", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        h, w, c = rgb_frame.shape
        qt_image = QImage(rgb_frame.data, w, h, 3 * w, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.width(), self.height(), Qt.KeepAspectRatio))

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        if self.landmarker:
            self.landmarker.close()
        super().closeEvent(event)


class WeatherIcon(QLabel):
    def __init__(self, weather_type, parent=None):
        super().__init__(parent)
        self.weather_type = weather_type
        self.setFixedSize(70, 70)
        self.setAlignment(Qt.AlignCenter)

        weather_styles = {
            '☀️ Güneşli':        ('☀️', '#FFD700', '#FFF8DC'),
            '🌧️ Yağmurlu':       ('🌧️', '#4682B4', '#E0F4FF'),
            '❄️ Karlı':          ('❄️', '#87CEEB', '#F0F8FF'),
            '⛅ Parçalı Bulutlu': ('⛅', '#B0C4DE', '#F5F5F5'),
            '🌩️ Fırtınalı':      ('🌩️', '#696969', '#DCDCDC'),
            '🌫️ Sisli':          ('🌫️', '#A9A9A9', '#F0F0F0'),
        }
        icon, bg, border = weather_styles[weather_type]
        self.setText(icon)
        self.base_style = f"""
            QLabel {{
                background-color: {bg}; border: 3px solid {border};
                border-radius: 14px; font-size: 42px; padding: 5px;
            }}
            QLabel:hover {{ border: 3px solid #FF6347; background-color: {border}; }}
        """
        self.setStyleSheet(self.base_style)
        self.setCursor(Qt.OpenHandCursor)
        self.is_selected = False

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            parent = self.parent()
            while parent and not isinstance(parent, TurkiyeHaritasiApp):
                parent = parent.parent()
            if parent:
                parent.select_weather(self.weather_type)

            self.setCursor(Qt.ClosedHandCursor)
            drag = QDrag(self)
            mime = QMimeData()
            mime.setText(self.weather_type)
            drag.setMimeData(mime)
            px = QPixmap(self.size())
            self.render(px)
            drag.setPixmap(px)
            drag.setHotSpot(event.pos())
            drag.exec_(Qt.CopyAction)
            self.setCursor(Qt.OpenHandCursor)

    def set_selected(self, selected):
        self.is_selected = selected
        if selected:
            self.setStyleSheet(self.base_style + """
                QLabel {
                    border: 4px solid #FF6347 !important;
                    background-color: rgba(255, 99, 71, 60) !important;
                }
            """)
        else:
            self.setStyleSheet(self.base_style)


class TurkiyeHaritasiApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.weather_data = {}
        self.selected_weather = None
        self.weather_icons = {}
        self.hand_mode_enabled = True
        self.is_grabbing = False

        self.photo_analyzer = PhotoAnalyzer()
        self.photo_analyzer.analysis_done.connect(self.on_photo_analyzed)
        self.photo_analyzer.status_changed.connect(self.on_voice_status)

        self.voice_worker = VoiceWorker()
        self.voice_worker.weather_detected.connect(self.on_voice_weather)
        self.voice_worker.region_detected.connect(self.zoom_to_region)
        self.voice_worker.status_changed.connect(self.on_voice_status)

        self.init_ui()

        if VOSK_AVAILABLE:
            self.voice_worker.start_listening()

    def init_ui(self):
        self.setWindowTitle('🇹🇷 Türkiye Hava Durumu Haritası')
        self.setGeometry(50, 50, 1600, 900)

        # Ana widget - harita tam ekran
        central = QWidget()
        self.setCentralWidget(central)
        central.setLayout(QHBoxLayout())
        central.layout().setContentsMargins(0, 0, 0, 0)

        # Harita full ekran
        self.web_view = QWebEngineView()
        self.web_view.setAcceptDrops(True)
        central.layout().addWidget(self.web_view)

        # Overlay panel - haritanın üzerinde, sol tarafta
        self.overlay = QWidget(central)
        self.overlay.setFixedWidth(200)
        self.overlay.setAttribute(Qt.WA_TranslucentBackground)
        overlay_layout = QVBoxLayout(self.overlay)
        overlay_layout.setContentsMargins(10, 10, 10, 10)
        overlay_layout.setSpacing(8)

        # Kamera widget
        self.camera_widget = CameraWidget(self.overlay)
        self.camera_widget.hand_position_changed.connect(self.handle_hand_gesture)
        self.camera_widget.setFixedSize(160, 120)
        overlay_layout.addWidget(self.camera_widget)

        # Ses durumu
        self.voice_label = QLabel('🎤 Dinleniyor...' if VOSK_AVAILABLE else '🎤 Kapalı')
        self.voice_label.setWordWrap(True)
        self.voice_label.setStyleSheet("""
            QLabel {
                background: rgba(142, 68, 173, 200);
                color: white; padding: 5px; border-radius: 8px;
                font-size: 10px; backdrop-filter: blur(10px);
            }
        """)
        overlay_layout.addWidget(self.voice_label)

        # Durum etiketi
        self.status_label = QLabel('👆 Seç veya söyle')
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("""
            QLabel {
                background: rgba(52, 152, 219, 200);
                color: white; padding: 5px; border-radius: 8px; font-size: 10px;
            }
        """)
        overlay_layout.addWidget(self.status_label)

        # Hava durumu ikonları - blur arka plan
        icons_bg = QWidget(self.overlay)
        icons_bg.setStyleSheet("""
            QWidget {
                background: rgba(30, 30, 50, 170);
                border-radius: 14px;
            }
        """)
        icons_layout = QGridLayout(icons_bg)
        icons_layout.setContentsMargins(8, 8, 8, 8)
        icons_layout.setSpacing(6)

        weather_types = ['☀️ Güneşli', '🌧️ Yağmurlu', '❄️ Karlı',
                         '⛅ Parçalı Bulutlu', '🌩️ Fırtınalı', '🌫️ Sisli']
        for idx, w in enumerate(weather_types):
            icon = WeatherIcon(w, self)
            self.weather_icons[w] = icon
            icon.setFixedSize(80, 80)
            icons_layout.addWidget(icon, idx // 2, idx % 2)
        overlay_layout.addWidget(icons_bg)

        # Fotoğraf yükle butonu
        photo_btn = QPushButton('📷 Fotoğraf Analiz Et')
        photo_btn.setStyleSheet("""
            QPushButton {
                background: rgba(22, 160, 133, 210);
                color: white; border: none; padding: 7px;
                font-size: 11px; font-weight: bold; border-radius: 8px;
            }
            QPushButton:hover { background: rgba(17, 124, 104, 230); }
        """)
        photo_btn.clicked.connect(self.open_photo)
        overlay_layout.addWidget(photo_btn)

        # Temizle butonu
        clear_btn = QPushButton('🗑️ Temizle')
        clear_btn.setStyleSheet("""
            QPushButton {
                background: rgba(231, 76, 60, 210);
                color: white; border: none; padding: 7px;
                font-size: 12px; font-weight: bold; border-radius: 8px;
            }
            QPushButton:hover { background: rgba(192, 57, 43, 230); }
        """)
        clear_btn.clicked.connect(self.clear_all_weather)
        overlay_layout.addWidget(clear_btn)

        overlay_layout.addStretch()
        self.overlay.move(0, 0)

        # Sağ üst köşe — bölge butonları overlay
        self.region_overlay = QWidget(central)
        self.region_overlay.setAttribute(Qt.WA_TranslucentBackground)
        region_layout = QVBoxLayout(self.region_overlay)
        region_layout.setContentsMargins(8, 8, 8, 8)
        region_layout.setSpacing(5)

        region_title = QLabel('📍 Bölgeler')
        region_title.setStyleSheet("QLabel { background: rgba(30,30,50,200); color:white; padding:5px 8px; border-radius:8px; font-size:11px; font-weight:bold; }")
        region_title.setAlignment(Qt.AlignCenter)
        region_layout.addWidget(region_title)

        self.regions = {
            'Marmara':      (40.8, 28.5, 9),
            'Ege':          (38.2, 27.8, 9),
            'Akdeniz':      (36.8, 31.0, 9),
            'İç Anadolu':   (39.0, 33.5, 8),
            'Karadeniz':    (41.3, 35.5, 9),
            'Doğu Anadolu': (39.2, 41.5, 8),
            'Güneydoğu':    (37.5, 39.5, 9),
        }

        for name in self.regions:
            btn = QPushButton(name)
            btn.setStyleSheet("""
                QPushButton {
                    background: rgba(44, 62, 80, 200);
                    color: white; border: none; padding: 6px 10px;
                    font-size: 11px; border-radius: 7px; text-align: left;
                }
                QPushButton:hover { background: rgba(52, 152, 219, 220); }
            """)
            btn.clicked.connect(lambda checked, n=name: self.zoom_to_region(n))
            region_layout.addWidget(btn)

        region_layout.addStretch()
        self.region_overlay.adjustSize()

        self.load_map()

    def on_voice_weather(self, weather_type, city):
        self.select_weather(weather_type)
        if city:
            # Direkt ile uygula
            self.apply_weather_to_city(weather_type, city)
            self.status_label.setText(f'🎤 {weather_type}\n📍 {city.title()}')
            self.status_label.setStyleSheet("QLabel { background: rgba(142,68,173,200); color:white; padding:5px; border-radius:8px; font-size:10px; }")
        else:
            self.status_label.setText(f'🎤 {weather_type}')
            self.status_label.setStyleSheet("QLabel { background: rgba(142,68,173,200); color:white; padding:5px; border-radius:8px; font-size:10px; }")

    def open_photo(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Fotoğraf Seç', '',
            'Resim Dosyaları (*.png *.jpg *.jpeg *.bmp *.webp)')
        if path:
            self.status_label.setText('🔍 Analiz ediliyor...')
            self.status_label.setStyleSheet(
                "QLabel { background: rgba(22,160,133,200); color:white; padding:5px; border-radius:8px; font-size:10px; }")
            self.photo_analyzer.analyze(path)

    def on_photo_analyzed(self, city, weather_type):
        if city and weather_type:
            self.select_weather(weather_type)
            self.apply_weather_to_city(weather_type, city)
            self.status_label.setText(f'📸 {city.title()} → {weather_type}')
            self.status_label.setStyleSheet(
                "QLabel { background: rgba(22,160,133,200); color:white; padding:5px; border-radius:8px; font-size:10px; }")
        elif weather_type and not city:
            self.select_weather(weather_type)
            self.status_label.setText(f'📸 {weather_type}\n(il bulunamadı)')
            self.status_label.setStyleSheet(
                "QLabel { background: rgba(243,156,18,200); color:white; padding:5px; border-radius:8px; font-size:10px; }")

    def on_voice_status(self, status):
        self.voice_label.setText(status)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'overlay'):
            self.overlay.setFixedHeight(self.centralWidget().height())
            self.overlay.move(0, 0)
        if hasattr(self, 'region_overlay'):
            self.region_overlay.adjustSize()
            w = self.centralWidget().width()
            self.region_overlay.move(w - self.region_overlay.width() - 10, 10)

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(100, lambda: self.resizeEvent(None))

    def apply_weather_to_city(self, weather_type, city_name):
        """Ses komutuyla il adına direkt hava durumu uygula"""
        weather_icon = weather_type.split()[0]
        weather_colors = {
            '☀️': '#FFD700', '🌧️': '#4682B4', '❄️': '#87CEEB',
            '⛅': '#B0C4DE', '🌩️': '#696969', '🌫️': '#A9A9A9'
        }
        fill_color = weather_colors.get(weather_icon, '#CCCCCC')
        search_name = city_name.lower()

        self.web_view.page().runJavaScript(f"""
        (function() {{
            if (typeof provinceLayer === 'undefined') return;
            var target = null;
            var search = '{search_name}';
            provinceLayer.eachLayer(function(layer) {{
                var name = (layer.feature.properties.name || '').toLowerCase();
                // Her iki yönde de kısmi eşleşme
                if (name.indexOf(search) !== -1 || search.indexOf(name) !== -1) {{
                    target = layer;
                }}
            }});
            if (target) {{
                target.setStyle({{
                    fillColor: '{fill_color}', fillOpacity: 0.6,
                    weight: 2, color: '#2C3E50'
                }});
                var center = target.getBounds().getCenter();
                if (target.weatherMarker) map.removeLayer(target.weatherMarker);
                target.weatherMarker = L.marker(center, {{
                    icon: L.divIcon({{
                        className: 'weather-icon',
                        html: '<div style="font-size:50px;text-shadow:2px 2px 4px rgba(0,0,0,0.5);">{weather_icon}</div>',
                        iconSize: [60,60], iconAnchor: [30,30]
                    }})
                }}).addTo(map);
                map.flyTo(center, 8, {{duration: 1.0}});
            }}
        }})();
        """)

    def zoom_to_region(self, name):
        lat, lng, zoom = self.regions[name]

        # Bölgeye göre il listesi
        region_cities = {
            'Marmara': ['istanbul', 'tekirdağ', 'edirne', 'kırklareli', 'çanakkale',
                        'balıkesir', 'bursa', 'yalova', 'kocaeli', 'sakarya', 'bilecik'],
            'Ege': ['izmir', 'manisa', 'aydın', 'denizli', 'muğla', 'uşak', 'kütahya', 'afyonkarahisar'],
            'Akdeniz': ['antalya', 'isparta', 'burdur', 'adana', 'mersin', 'hatay',
                        'kahramanmaraş', 'osmaniye'],
            'İç Anadolu': ['ankara', 'konya', 'eskişehir', 'kayseri', 'sivas', 'yozgat',
                           'kırıkkale', 'kırşehir', 'nevşehir', 'aksaray', 'niğde', 'karaman'],
            'Karadeniz': ['zonguldak', 'bartın', 'karabük', 'bolu', 'düzce', 'sakarya',
                          'sinop', 'kastamonu', 'çankırı', 'samsun', 'ordu', 'giresun',
                          'trabzon', 'rize', 'artvin', 'amasya', 'tokat', 'gümüşhane', 'bayburt'],
            'Doğu Anadolu': ['erzurum', 'erzincan', 'ağrı', 'kars', 'ardahan', 'iğdır',
                              'elazığ', 'malatya', 'bingöl', 'tunceli', 'muş', 'bitlis',
                              'van', 'hakkari'],
            'Güneydoğu': ['gaziantep', 'adıyaman', 'şanlıurfa', 'diyarbakır', 'mardin',
                           'batman', 'siirt', 'şırnak', 'kilis'],
        }

        cities = region_cities.get(name, [])
        cities_json = json.dumps(cities)

        self.web_view.page().runJavaScript(f"""
        (function() {{
            var regionCities = {cities_json};
            var bounds = null;
            provinceLayer.eachLayer(function(layer) {{
                var pname = (layer.feature.properties.name || '').toLowerCase();
                var inRegion = regionCities.some(function(c) {{
                    return pname.indexOf(c) !== -1 || c.indexOf(pname) !== -1;
                }});
                if (!inRegion) {{
                    layer.setStyle({{
                        fillColor: '#888888',
                        fillOpacity: 0.45,
                        color: '#555555',
                        weight: 1
                    }});
                }} else {{
                    if (!layer.options._hasWeather) {{
                        layer.setStyle({{
                            fillColor: 'transparent',
                            fillOpacity: 0.1,
                            color: '#2C3E50',
                            weight: 2
                        }});
                    }}
                    var lb = layer.getBounds();
                    if (!bounds) {{
                        bounds = lb;
                    }} else {{
                        bounds.extend(lb);
                    }}
                }}
            }});
            if (bounds) {{
                map.flyToBounds(bounds, {{padding: [30, 30], duration: 1.2}});
            }}
        }})();
        """)

    def toggle_hand_mode(self, state):
        self.hand_mode_enabled = state == Qt.Checked

    def select_weather(self, weather_type):
        for icon in self.weather_icons.values():
            icon.set_selected(False)
        self.selected_weather = weather_type
        self.weather_icons[weather_type].set_selected(True)
        self.status_label.setText(f'✅ {weather_type}\n✊ Pinch ile yerleştir')
        self.status_label.setStyleSheet("QLabel { background: rgba(39,174,96,200); color:white; padding:5px; border-radius:8px; font-size:10px; }")

    def handle_hand_gesture(self, norm_x, norm_y, is_pinching):
        if not self.hand_mode_enabled:
            return

        web_w = self.web_view.width()
        web_h = self.web_view.height()
        map_x = int(norm_x * web_w)
        map_y = int(norm_y * web_h)

        # Elin ekran pozisyonu (pencereye göre)
        central = self.centralWidget()
        hand_screen_x = int(norm_x * central.width())
        hand_screen_y = int(norm_y * central.height())

        # El ikonların üzerinde mi?
        hovered_icon = None
        for weather_type, icon in self.weather_icons.items():
            # İkonun merkez widget'a göre pozisyonu
            icon_pos = icon.mapTo(central, QPoint(0, 0))
            if (icon_pos.x() <= hand_screen_x <= icon_pos.x() + icon.width() and
                    icon_pos.y() <= hand_screen_y <= icon_pos.y() + icon.height()):
                hovered_icon = weather_type
                break

        if hovered_icon:
            # El ikonu üzerinde — pinch ile seç
            if is_pinching and not self.is_grabbing:
                self.is_grabbing = True
                self.select_weather(hovered_icon)
            if not is_pinching:
                self.is_grabbing = False
            return  # harita marker'ı güncelleme

        # El harita üzerinde — marker göster
        self.web_view.page().runJavaScript(f"""
        (function() {{
            if (typeof map === 'undefined') return;
            var latlng = map.containerPointToLatLng([{map_x}, {map_y}]);
            if (window._handMarker) {{
                window._handMarker.setLatLng(latlng);
            }} else {{
                window._handMarker = L.circleMarker(latlng, {{
                    radius: 14, color: '#FF0000', weight: 3,
                    fillColor: '#FF4444', fillOpacity: 0.85, interactive: false
                }}).addTo(map);
            }}
        }})();
        """)

        if self.selected_weather:
            if is_pinching and not self.is_grabbing:
                self.is_grabbing = True
                self.status_label.setText(f'✊ {self.selected_weather}')
                self.status_label.setStyleSheet("QLabel { background: rgba(243,156,18,200); color:white; padding:5px; border-radius:8px; font-size:10px; }")

            if is_pinching and self.is_grabbing:
                # Pinch devam ederken pozisyonu kaydet
                self.last_pinch_x = map_x
                self.last_pinch_y = map_y

            elif not is_pinching and self.is_grabbing:
                self.is_grabbing = False
                # Son pinch pozisyonunu kullan, şu anki değil
                use_x = getattr(self, 'last_pinch_x', map_x)
                use_y = getattr(self, 'last_pinch_y', map_y)
                self.place_weather_on_map(use_x, use_y)
                self.status_label.setText('✅ Yerleştirildi!')
                self.status_label.setStyleSheet("QLabel { background: rgba(52,152,219,200); color:white; padding:5px; border-radius:8px; font-size:10px; }")

        if not is_pinching:
            self.is_grabbing = False

    def place_weather_on_map(self, map_x, map_y):
        if not self.selected_weather:
            return

        weather_icon = self.selected_weather.split()[0]
        weather_colors = {
            '☀️': '#FFD700', '🌧️': '#4682B4', '❄️': '#87CEEB',
            '⛅': '#B0C4DE', '🌩️': '#696969', '🌫️': '#A9A9A9'
        }
        fill_color = weather_colors.get(weather_icon, '#CCCCCC')

        self.web_view.page().runJavaScript(f"""
        (function() {{
            if (typeof map === 'undefined' || typeof provinceLayer === 'undefined') return;
            var point = map.containerPointToLatLng([{map_x}, {map_y}]);
            var closest = null, minDist = Infinity;
            provinceLayer.eachLayer(function(layer) {{
                var d = point.distanceTo(layer.getBounds().getCenter());
                if (d < minDist) {{ minDist = d; closest = layer; }}
            }});
            if (closest) {{
                closest.setStyle({{ fillColor: '{fill_color}', fillOpacity: 0.6, weight: 2, color: '#2C3E50' }});
                closest.options._hasWeather = true;
                var center = closest.getBounds().getCenter();
                if (closest.weatherMarker) map.removeLayer(closest.weatherMarker);
                closest.weatherMarker = L.marker(center, {{
                    icon: L.divIcon({{
                        className: 'weather-icon',
                        html: '<div style="font-size:50px;text-shadow:2px 2px 4px rgba(0,0,0,0.5);">{weather_icon}</div>',
                        iconSize: [60,60], iconAnchor: [30,30]
                    }})
                }}).addTo(map);
            }}
        }})();
        """)

    def load_map(self):
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
            <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
            <style>
                body { margin:0; padding:0; }
                #map { width:100%; height:100vh; }
            </style>
        </head>
        <body>
        <div id="map"></div>
        <script>
            var map = L.map('map').setView([39.0, 35.0], 6);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '© OpenStreetMap contributors', maxZoom: 18
            }).addTo(map);

            var provinceLayer = L.layerGroup().addTo(map);

            fetch('https://raw.githubusercontent.com/cihadturhan/tr-geojson/master/geo/tr-cities-utf8.json')
                .then(r => r.json())
                .then(data => {
                    L.geoJSON(data, {
                        style: () => ({ fillColor: 'transparent', weight: 2, color: '#2C3E50', fillOpacity: 0.1 }),
                        onEachFeature: function(feature, layer) {
                            layer.on({
                                mouseover: e => {
                                    if (!e.target.options.fillColor || e.target.options.fillColor === 'transparent')
                                        e.target.setStyle({ fillOpacity: 0.3, fillColor: '#3498DB' });
                                },
                                mouseout: e => {
                                    if (e.target.options.fillOpacity === 0.3)
                                        e.target.setStyle({ fillOpacity: 0.1, fillColor: 'transparent' });
                                }
                            });
                            layer.bindTooltip(feature.properties.name, { permanent: false, direction: 'center' });
                            provinceLayer.addLayer(layer);
                        }
                    });
                })
                .catch(() => alert('Harita verileri yüklenemedi.'));
        </script>
        </body>
        </html>
        """
        self.web_view.setHtml(html)
        self.web_view.dragEnterEvent = self.drag_enter_event
        self.web_view.dropEvent = self.drop_event

    def drag_enter_event(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def drop_event(self, event):
        weather_type = event.mimeData().text()
        weather_icon = weather_type.split()[0]
        weather_colors = {
            '☀️': '#FFD700', '🌧️': '#4682B4', '❄️': '#87CEEB',
            '⛅': '#B0C4DE', '🌩️': '#696969', '🌫️': '#A9A9A9'
        }
        fill_color = weather_colors.get(weather_icon, '#CCCCCC')
        pos = event.pos()

        self.web_view.page().runJavaScript(f"""
        (function() {{
            var point = map.containerPointToLatLng([{pos.x()}, {pos.y()}]);
            var closest = null, minDist = Infinity;
            provinceLayer.eachLayer(function(layer) {{
                var d = point.distanceTo(layer.getBounds().getCenter());
                if (d < minDist) {{ minDist = d; closest = layer; }}
            }});
            if (closest) {{
                closest.setStyle({{ fillColor: '{fill_color}', fillOpacity: 0.6, weight: 2, color: '#2C3E50' }});
                var center = closest.getBounds().getCenter();
                if (closest.weatherMarker) map.removeLayer(closest.weatherMarker);
                closest.weatherMarker = L.marker(center, {{
                    icon: L.divIcon({{
                        className: 'weather-icon',
                        html: '<div style="font-size:50px;text-shadow:2px 2px 4px rgba(0,0,0,0.5);">{weather_icon}</div>',
                        iconSize: [60,60], iconAnchor: [30,30]
                    }})
                }}).addTo(map);
            }}
        }})();
        """)
        event.acceptProposedAction()

    def clear_all_weather(self):
        self.weather_data.clear()
        self.selected_weather = None
        for icon in self.weather_icons.values():
            icon.set_selected(False)
        self.web_view.page().runJavaScript("""
        provinceLayer.eachLayer(function(layer) {
            layer.setStyle({ fillColor: 'transparent', fillOpacity: 0.1, weight: 2, color: '#2C3E50' });
            layer.options._hasWeather = false;
            if (layer.weatherMarker) { map.removeLayer(layer.weatherMarker); delete layer.weatherMarker; }
        });
        """)
        self.status_label.setText('✅ Temizlendi!')
        self.status_label.setStyleSheet("QLabel { background: rgba(52,152,219,200); color:white; padding:5px; border-radius:8px; font-size:10px; }")

    def closeEvent(self, event):
        self.voice_worker.stop_listening()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    print("=" * 50)
    print("🇹🇷 Türkiye Hava Durumu Haritası")
    print("=" * 50)
    print(f"MediaPipe:         {'✅' if MEDIAPIPE_AVAILABLE else '❌'}")
    print(f"Vosk:              {'✅' if VOSK_AVAILABLE else '❌'}")
    print("=" * 50)
    print("Sesli komutlar: güneşli, yağmurlu, karlı, bulutlu, fırtınalı, sisli")
    print("=" * 50)
    window = TurkiyeHaritasiApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()