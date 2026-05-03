import sys
import json
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QComboBox, QCheckBox, QGridLayout)
from PyQt5.QtCore import Qt, QUrl, QMimeData, QPoint, QTimer, pyqtSignal, QRect
from PyQt5.QtGui import QDrag, QPainter, QColor, QFont, QPixmap, QImage
from PyQt5.QtWebEngineWidgets import QWebEngineView

try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
except Exception as e:
    print(f"MediaPipe yüklenemedi: {e}")
    MEDIAPIPE_AVAILABLE = False


class CameraWidget(QLabel):
    hand_position_changed = pyqtSignal(int, int, bool)  # x, y, is_pinching

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

        # MediaPipe el algılama
        if MEDIAPIPE_AVAILABLE:
            try:
                self.mp_hands = mp.solutions.hands
                self.mp_draw = mp.solutions.drawing_utils
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7
                )
                self.mediapipe_ok = True
                print("✅ MediaPipe başarıyla yüklendi!")
            except Exception as e:
                print(f"❌ MediaPipe başlatılamadı: {e}")
                self.mediapipe_ok = False

        # Kamera
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

        self.is_pinching = False
        self.prev_pinching = False

    def show_error_message(self, message):
        self.setText(f"⚠️\n{message}")
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 3px solid #E74C3C;
                border-radius: 10px;
                background-color: #2C3E50;
                color: white;
                font-size: 14px;
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
        h, w, c = rgb_frame.shape

        # El algılama (sadece MediaPipe varsa)
        if self.mediapipe_ok:
            try:
                results = self.hands.process(rgb_frame)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # El çiz
                        self.mp_draw.draw_landmarks(
                            rgb_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                        # Başparmak ve işaret parmağı uçları
                        thumb_tip = hand_landmarks.landmark[4]
                        index_tip = hand_landmarks.landmark[8]

                        thumb_x = int(thumb_tip.x * w)
                        thumb_y = int(thumb_tip.y * h)
                        index_x = int(index_tip.x * w)
                        index_y = int(index_tip.y * h)

                        # Parmaklar arası mesafe
                        distance = np.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)

                        # Pinch hareketi kontrolü
                        self.is_pinching = distance < 40

                        # İşaret parmağı pozisyonu
                        screen_x = int(index_tip.x * self.width())
                        screen_y = int(index_tip.y * self.height())

                        # Görsel feedback
                        if self.is_pinching:
                            cv2.circle(rgb_frame, (index_x, index_y), 15, (0, 255, 0), -1)
                            cv2.line(rgb_frame, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 0), 3)
                            cv2.putText(rgb_frame, "TUTTU!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        else:
                            cv2.circle(rgb_frame, (index_x, index_y), 10, (255, 0, 0), 2)

                        # Sinyal gönder
                        self.hand_position_changed.emit(screen_x, screen_y, self.is_pinching)
            except Exception as e:
                print(f"El algılama hatası: {e}")
        else:
            # MediaPipe yoksa uyarı göster
            cv2.putText(rgb_frame, "MediaPipe yuklenemedi", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(rgb_frame, "Fare ile kullanin", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # QLabel'a göster
        h, w, c = rgb_frame.shape
        bytes_per_line = 3 * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.width(), self.height(), Qt.KeepAspectRatio))

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        super().closeEvent(event)


class WeatherIcon(QLabel):
    def __init__(self, weather_type, parent=None):
        super().__init__(parent)
        self.weather_type = weather_type
        self.setFixedSize(70, 70)
        self.setAlignment(Qt.AlignCenter)

        weather_styles = {
            '☀️ Güneşli': ('☀️', '#FFD700', '#FFF8DC'),
            '🌧️ Yağmurlu': ('🌧️', '#4682B4', '#E0F4FF'),
            '❄️ Karlı': ('❄️', '#87CEEB', '#F0F8FF'),
            '⛅ Parçalı Bulutlu': ('⛅', '#B0C4DE', '#F5F5F5'),
            '🌩️ Fırtınalı': ('🌩️', '#696969', '#DCDCDC'),
            '🌫️ Sisli': ('🌫️', '#A9A9A9', '#F0F0F0')
        }

        icon, bg_color, border_color = weather_styles[weather_type]

        self.setText(icon)
        self.base_style = f"""
            QLabel {{
                background-color: {bg_color};
                border: 3px solid {border_color};
                border-radius: 10px;
                font-size: 35px;
                padding: 5px;
            }}
            QLabel:hover {{
                border: 3px solid #FF6347;
                background-color: {border_color};
            }}
        """
        self.setStyleSheet(self.base_style)

        self.setCursor(Qt.OpenHandCursor)
        self.is_selected = False

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Hem sürükle-bırak hem de seçim için
            parent = self.parent()
            while parent and not isinstance(parent, TurkiyeHaritasiApp):
                parent = parent.parent()

            if parent:
                parent.select_weather_for_hand(self.weather_type)

            # Drag başlat
            self.setCursor(Qt.ClosedHandCursor)
            drag = QDrag(self)
            mime_data = QMimeData()
            mime_data.setText(self.weather_type)
            drag.setMimeData(mime_data)

            pixmap = QPixmap(self.size())
            self.render(pixmap)
            drag.setPixmap(pixmap)
            drag.setHotSpot(event.pos())

            drag.exec_(Qt.CopyAction)
            self.setCursor(Qt.OpenHandCursor)

    def set_selected(self, selected):
        self.is_selected = selected
        if selected:
            self.setStyleSheet(self.base_style + """
                QLabel { border: 4px solid #FF6347 !important; }
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
        self.init_ui()

    def update_pointer_on_map(self, x, y):
        web_rect = self.web_view.geometry()
        relative_x = (x / self.camera_widget.width()) * web_rect.width()
        relative_y = (y / self.camera_widget.height()) * web_rect.height()

        js_code = f"""
        (function() {{
            if (window.handPointer) {{
                map.removeLayer(window.handPointer);
            }}
            var containerPoint = [{relative_x}, {relative_y}];
            var point = map.containerPointToLatLng(containerPoint);

            window.handPointer = L.circleMarker(point, {{
                radius: 10,
                color: 'red',
                fillColor: '#f03',
                fillOpacity: 0.8
            }}).addTo(map);
        }})();
        """
        self.web_view.page().runJavaScript(js_code)

    def init_ui(self):
        self.setWindowTitle('🇹🇷 Türkiye Hava Durumu Haritası - El Hareketli Kontrol')
        self.setGeometry(50, 50, 1600, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Sol panel
        left_panel = QWidget()
        left_panel.setMaximumWidth(300)
        left_layout = QVBoxLayout(left_panel)

        # Kamera widget
        cam_title = QLabel('📷 Kamera (El Hareketi)')
        cam_title.setStyleSheet("""
            QLabel {
                font-size: 13px;
                font-weight: bold;
                padding: 5px;
                background-color: #34495E;
                color: white;
                border-radius: 3px;
            }
        """)
        left_layout.addWidget(cam_title)

        self.camera_widget = CameraWidget()
        self.camera_widget.hand_position_changed.connect(self.handle_hand_gesture)
        left_layout.addWidget(self.camera_widget)

        # El modu toggle
        self.hand_mode_check = QCheckBox('✋ El Modu Aktif')
        self.hand_mode_check.setChecked(True)
        self.hand_mode_check.stateChanged.connect(self.toggle_hand_mode)
        self.hand_mode_check.setStyleSheet("""
            QCheckBox {
                font-size: 14px;
                font-weight: bold;
                padding: 5px;
            }
        """)
        left_layout.addWidget(self.hand_mode_check)

        # Durum göstergesi
        self.status_label = QLabel('👆 Hava durumu seçin ve parmakları birleştirin')
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #3498DB;
                color: white;
                padding: 8px;
                border-radius: 5px;
                font-size: 11px;
            }
        """)
        left_layout.addWidget(self.status_label)

        # Hava durumu başlığı
        title = QLabel('🌍 Hava Durumu Simgeleri')
        title.setStyleSheet("""
            QLabel {
                font-size: 15px;
                font-weight: bold;
                padding: 10px;
                background-color: #2C3E50;
                color: white;
                border-radius: 5px;
            }
        """)
        left_layout.addWidget(title)

        # Hava durumu simgeleri için grid layout
        weather_container = QWidget()
        weather_grid = QGridLayout(weather_container)
        weather_grid.setSpacing(5)

        weather_types = ['☀️ Güneşli', '🌧️ Yağmurlu', '❄️ Karlı',
                         '⛅ Parçalı Bulutlu', '🌩️ Fırtınalı', '🌫️ Sisli']

        for idx, weather in enumerate(weather_types):
            icon = WeatherIcon(weather, self)
            self.weather_icons[weather] = icon
            row = idx // 2
            col = idx % 2
            weather_grid.addWidget(icon, row, col)

        left_layout.addWidget(weather_container)
        left_layout.addStretch()

        # Temizle butonu
        clear_btn = QPushButton('🗑️ Tümünü Temizle')
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #E74C3C;
                color: white;
                border: none;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #C0392B;
            }
        """)
        clear_btn.clicked.connect(self.clear_all_weather)
        left_layout.addWidget(clear_btn)

        # Sağ panel - Harita
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.web_view = QWebEngineView()
        self.web_view.setAcceptDrops(True)
        right_layout.addWidget(self.web_view)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, stretch=1)

        self.load_map()

    def toggle_hand_mode(self, state):
        self.hand_mode_enabled = state == Qt.Checked

    def select_weather_for_hand(self, weather_type):
        # Tüm iconları deselect
        for icon in self.weather_icons.values():
            icon.set_selected(False)

        # Seçili olanı işaretle
        self.selected_weather = weather_type
        self.weather_icons[weather_type].set_selected(True)

        if self.hand_mode_enabled:
            self.status_label.setText(f'✅ Seçili: {weather_type}\n👉 Parmakları birleştirip sürükleyin')
            self.status_label.setStyleSheet("""
                QLabel {
                    background-color: #27AE60;
                    color: white;
                    padding: 8px;
                    border-radius: 5px;
                    font-size: 11px;
                }
            """)

    def check_hand_over_icon(self, x, y):
        """Elin hangi ikonun üzerinde olduğunu kontrol et"""
        camera_rect = self.camera_widget.geometry()

        for weather_type, icon in self.weather_icons.items():
            icon_rect = icon.geometry()
            icon_y_start = icon.mapTo(self.camera_widget.parent(), QPoint(0, 0)).y()
            icon_y_end = icon_y_start + icon.height()
            icon_x_start = icon.mapTo(self.camera_widget.parent(), QPoint(0, 0)).x()
            icon_x_end = icon_x_start + icon.width()

            camera_to_panel = self.camera_widget.mapTo(self.camera_widget.parent(), QPoint(x, y))

            if (icon_x_start <= camera_to_panel.x() <= icon_x_end and
                    icon_y_start <= camera_to_panel.y() <= icon_y_end):
                return weather_type

        return None

    def handle_hand_gesture(self, x, y, is_pinching):
        if not self.hand_mode_enabled:
            return

        # İkonların üzerinde mi kontrol et
        hovered_icon = self.check_hand_over_icon(x, y)

        if hovered_icon:
            # İkon üzerinde - pinch ile seç
            if is_pinching and not self.is_grabbing:
                self.select_weather_for_hand(hovered_icon)
                self.is_grabbing = True
        elif self.selected_weather:
            # Harita üzerinde - pointer göster
            self.update_pointer_on_map(x, y)

            # Pinch hareketi ile bırakma
            if is_pinching and not self.is_grabbing:
                self.is_grabbing = True
                self.status_label.setText(f'✊ Tutuyorsunuz: {self.selected_weather}')
                self.status_label.setStyleSheet("""
                    QLabel {
                        background-color: #F39C12;
                        color: white;
                        padding: 8px;
                        border-radius: 5px;
                        font-size: 11px;
                    }
                """)
            elif not is_pinching and self.is_grabbing:
                self.is_grabbing = False
                # Haritaya bırak
                self.drop_weather_on_map(x, y)
                self.status_label.setText(f'✅ Bırakıldı!\n👆 Başka hava durumu seçebilirsiniz')
                self.status_label.setStyleSheet("""
                    QLabel {
                        background-color: #3498DB;
                        color: white;
                        padding: 8px;
                        border-radius: 5px;
                        font-size: 11px;
                    }
                """)

        # Pinch bırakıldığında grabbing durumunu sıfırla
        if not is_pinching:
            self.is_grabbing = False

    def drop_weather_on_map(self, x, y):
        if not self.selected_weather:
            return

        weather_icon = self.selected_weather.split()[0]

        # Hava durumu renklerini al
        weather_colors = {
            '☀️': '#FFD700',  # Güneşli - Altın sarısı
            '🌧️': '#4682B4',  # Yağmurlu - Mavi
            '❄️': '#87CEEB',  # Karlı - Açık mavi
            '⛅': '#B0C4DE',  # Parçalı bulutlu - Gri-mavi
            '🌩️': '#696969',  # Fırtınalı - Koyu gri
            '🌫️': '#A9A9A9'  # Sisli - Gri
        }

        fill_color = weather_colors.get(weather_icon, '#CCCCCC')

        # Web view koordinatlarına göre ayarla
        web_rect = self.web_view.geometry()
        relative_x = (x / self.camera_widget.width()) * web_rect.width()
        relative_y = (y / self.camera_widget.height()) * web_rect.height()

        js_code = f"""
        (function() {{
            var containerPoint = [{relative_x}, {relative_y}];
            var point = map.containerPointToLatLng(containerPoint);

            var closestProvince = null;
            var minDistance = Infinity;

            // GeoJSON layer'ları içinde ara
            provinceLayer.eachLayer(function(layer) {{
                var bounds = layer.getBounds();
                var center = bounds.getCenter();
                var distance = point.distanceTo(center);

                if (distance < minDistance) {{
                    minDistance = distance;
                    closestProvince = layer;
                }}
            }});

            if (closestProvince) {{
                var provinceName = closestProvince.feature.properties.name;

                // İl sınırlarını renklendir ve ikonu ekle
                closestProvince.setStyle({{
                    fillColor: '{fill_color}',
                    fillOpacity: 0.6,
                    weight: 2,
                    color: '#2C3E50',
                    dashArray: ''
                }});

                // İl merkezine emoji ekle
                var center = closestProvince.getBounds().getCenter();

                // Eski marker varsa kaldır
                if (closestProvince.weatherMarker) {{
                    map.removeLayer(closestProvince.weatherMarker);
                }}

                // Yeni marker ekle
                closestProvince.weatherMarker = L.marker(center, {{
                    icon: L.divIcon({{
                        className: 'weather-province-icon',
                        html: '<div style="font-size: 50px; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">{weather_icon}</div>',
                        iconSize: [60, 60],
                        iconAnchor: [30, 30]
                    }})
                }}).addTo(map);

                return provinceName;
            }}
            return null;
        }})();
        """

        self.web_view.page().runJavaScript(js_code)

    def load_map(self):
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
            <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
            <style>
                body { margin: 0; padding: 0; }
                #map { width: 100%; height: 100vh; }
            </style>
        </head>
        <body>
            <div id="map"></div>
            <script>
                var map = L.map('map').setView([39.0, 35.0], 6);

                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '© OpenStreetMap contributors',
                    maxZoom: 18
                }).addTo(map);

                var provinceLayer = L.layerGroup().addTo(map);

                // Türkiye il sınırları GeoJSON (basitleştirilmiş)
                fetch('https://raw.githubusercontent.com/cihadturhan/tr-geojson/master/geo/tr-cities-utf8.json')
                    .then(response => response.json())
                    .then(data => {
                        L.geoJSON(data, {
                            style: function(feature) {
                                return {
                                    fillColor: 'transparent',
                                    weight: 2,
                                    opacity: 1,
                                    color: '#2C3E50',
                                    fillOpacity: 0.1
                                };
                            },
                            onEachFeature: function(feature, layer) {
                                layer.on({
                                    mouseover: function(e) {
                                        if (!e.target.options.fillColor || e.target.options.fillColor === 'transparent') {
                                            e.target.setStyle({
                                                fillOpacity: 0.3,
                                                fillColor: '#3498DB'
                                            });
                                        }
                                    },
                                    mouseout: function(e) {
                                        if (!e.target.options.fillColor || e.target.options.fillColor === 'transparent' || e.target.options.fillOpacity === 0.3) {
                                            e.target.setStyle({
                                                fillOpacity: 0.1,
                                                fillColor: 'transparent'
                                            });
                                        }
                                    }
                                });

                                // Tooltip ekle
                                layer.bindTooltip(feature.properties.name, {
                                    permanent: false,
                                    direction: 'center',
                                    className: 'province-tooltip'
                                });

                                provinceLayer.addLayer(layer);
                            }
                        });
                    })
                    .catch(error => {
                        console.error('GeoJSON yüklenemedi:', error);
                        alert('Harita verileri yüklenemedi. İnternet bağlantınızı kontrol edin.');
                    });
            </script>
        </body>
        </html>
        """

        self.web_view.setHtml(html_content)
        self.web_view.dragEnterEvent = self.drag_enter_event
        self.web_view.dropEvent = self.drop_event

    def drag_enter_event(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def drop_event(self, event):
        weather_type = event.mimeData().text()
        weather_icon = weather_type.split()[0]

        # Hava durumu renkleri
        weather_colors = {
            '☀️': '#FFD700',
            '🌧️': '#4682B4',
            '❄️': '#87CEEB',
            '⛅': '#B0C4DE',
            '🌩️': '#696969',
            '🌫️': '#A9A9A9'
        }

        fill_color = weather_colors.get(weather_icon, '#CCCCCC')
        pos = event.pos()

        js_code = f"""
        (function() {{
            var point = map.containerPointToLatLng([{pos.x()}, {pos.y()}]);

            var closestProvince = null;
            var minDistance = Infinity;

            provinceLayer.eachLayer(function(layer) {{
                var bounds = layer.getBounds();
                var center = bounds.getCenter();
                var distance = point.distanceTo(center);

                if (distance < minDistance) {{
                    minDistance = distance;
                    closestProvince = layer;
                }}
            }});

            if (closestProvince) {{
                closestProvince.setStyle({{
                    fillColor: '{fill_color}',
                    fillOpacity: 0.6,
                    weight: 2,
                    color: '#2C3E50',
                    dashArray: ''
                }});

                var center = closestProvince.getBounds().getCenter();

                if (closestProvince.weatherMarker) {{
                    map.removeLayer(closestProvince.weatherMarker);
                }}

                closestProvince.weatherMarker = L.marker(center, {{
                    icon: L.divIcon({{
                        className: 'weather-province-icon',
                        html: '<div style="font-size: 50px; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">{weather_icon}</div>',
                        iconSize: [60, 60],
                        iconAnchor: [30, 30]
                    }})
                }}).addTo(map);
            }}
        }})();
        """

        self.web_view.page().runJavaScript(js_code)
        event.acceptProposedAction()

    def clear_all_weather(self):
        self.weather_data.clear()
        self.selected_weather = None
        for icon in self.weather_icons.values():
            icon.set_selected(False)

        # Tüm il renklerini ve marker'ları temizle
        js_code = """
        provinceLayer.eachLayer(function(layer) {
            layer.setStyle({
                fillColor: 'transparent',
                fillOpacity: 0.1,
                weight: 2,
                color: '#2C3E50'
            });

            if (layer.weatherMarker) {
                map.removeLayer(layer.weatherMarker);
                delete layer.weatherMarker;
            }
        });
        """
        self.web_view.page().runJavaScript(js_code)

        self.status_label.setText('✅ Temizlendi! Yeni hava durumu seçin')
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #3498DB;
                color: white;
                padding: 8px;
                border-radius: 5px;
                font-size: 11px;
            }
        """)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    print("=" * 50)
    print("🇹🇷 Türkiye Hava Durumu Haritası")
    print("=" * 50)
    print(f"MediaPipe: {'✅ Yüklendi' if MEDIAPIPE_AVAILABLE else '❌ Yüklenemedi'}")
    print("=" * 50)

    window = TurkiyeHaritasiApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()