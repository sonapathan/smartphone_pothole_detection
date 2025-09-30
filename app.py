from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserIconView
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.utils import get_color_from_hex
from kivy.uix.behaviors import ButtonBehavior
from kivy.graphics import Color, RoundedRectangle

import cv2
import pyttsx3
from time import time
from ultralytics import YOLO
from plyer import gps, accelerometer
import threading
import speech_recognition as sr

# Window settings
Window.size = (360, 640)
Window.clearcolor = get_color_from_hex("#E7F0F2")


# Utility Functions
def is_night_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()
    print(f"Image brightness: {brightness}")
    return brightness < 80


def image_to_texture(img):
    img = cv2.flip(img, 0)
    buf = img.tobytes()
    texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='bgr')
    texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
    return texture


# Custom Rounded Button
class RoundedButton(ButtonBehavior, BoxLayout):
    def __init__(self, text='', **kwargs):
        super().__init__(orientation='horizontal', padding=10, spacing=10, **kwargs)
        self.text = text
        self.bg_color = get_color_from_hex("#042232")
        self.pressed_color = get_color_from_hex("#10010D")
        self.current_color = self.bg_color

        self.bind(pos=self.update_canvas, size=self.update_canvas)
        with self.canvas.before:
            Color(*self.current_color)
            self.bg_rect = RoundedRectangle(pos=self.pos, size=self.size, radius=[20])

        self.label = Label(text=self.text, color=(1, 1, 1, 1), font_size='16sp', bold=True)
        self.add_widget(self.label)

        self.bind(on_press=self.on_press_event, on_release=self.on_release_event)

    def update_canvas(self, *args):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(*self.current_color)
            self.bg_rect = RoundedRectangle(pos=self.pos, size=self.size, radius=[20])

    def on_press_event(self, *args):
        self.current_color = self.pressed_color
        self.update_canvas()

    def on_release_event(self, *args):
        self.current_color = self.bg_color
        self.update_canvas()


# Main Layout
class MainLayout(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', padding=15, spacing=15, **kwargs)
        self.capture = None
        self.live_only = False
        self.engine = pyttsx3.init()
        self.last_spoken = 0
        self.speak_cooldown = 5
        self.last_detection_status = None

        # Load both models
        self.model_day = YOLO(r'C:\Users\PMLS\Downloads\Desktop\thesis\finalyearproject\yolov8_model\model-bestpt\daypothole.pt')
        self.model_night = YOLO(r'C:\Users\PMLS\Downloads\Desktop\thesis\finalyearproject\yolov8_model\model-bestpt\nightpothole.pt')

        self.init_tts()
        self.start_gps()
        self.start_accelerometer()
        threading.Thread(target=self.listen_voice_commands, daemon=True).start()

        # --- Fixed Small Heading ---
        header = Label(
            text='[b]\U0001F6A7 Smartphone Detection System Using YOLOv8 for Real-Time '
                 'Day and Night Road Condition Monitoring[/b]',
            markup=True,
            size_hint=(1, None),   # Fixed height
            height=70,             # Enough for 2-line wrap
            font_size='14sp',      # Smaller font so long text fits
            color=get_color_from_hex("#021E34"),
            halign="center",
            valign="middle"
        )
        header.bind(size=header.setter('text_size'))  # Enable wrapping

        # --- Widgets ---
        self.image_view = Image(size_hint=(1, 0.5))

        self.status_button = Label(text='Status: Unknown', size_hint=(1, 0.1), font_size='18sp', bold=True,
                                   color=(1, 1, 1, 1))
        self.update_status_button("Neutral")

        btn_upload = RoundedButton(text='Upload Image', size_hint=(1, 0.1))
        btn_camera = RoundedButton(text='Capture from Camera', size_hint=(1, 0.1))
        btn_video = RoundedButton(text='Start Video Detection', size_hint=(1, 0.1))
        btn_live = RoundedButton(text='Live Detection', size_hint=(1, 0.1))

        btn_upload.bind(on_press=self.open_filechooser)
        btn_camera.bind(on_press=self.capture_camera)
        btn_video.bind(on_press=self.start_video_stream)
        btn_live.bind(on_press=self.start_live_stream)

        # Add widgets to layout
        self.add_widget(header)
        self.add_widget(self.image_view)
        self.add_widget(self.status_button)
        self.add_widget(btn_upload)
        self.add_widget(btn_camera)
        self.add_widget(btn_video)
        self.add_widget(btn_live)

    # --- Object Detection ---
    def detect_objects(self, frame):
        temp_path = 'temp_input.jpg'
        cv2.imwrite(temp_path, frame)

        model_to_use = self.model_night if is_night_image(frame) else self.model_day
        print("Model used:", "Night" if model_to_use == self.model_night else "Day")
        results = model_to_use(temp_path, save=False)

        detections = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = results[0].names[cls_id]
                detections.append(label)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        return frame, detections

    # --- TTS ---
    def init_tts(self):
        voices = self.engine.getProperty('voices')
        for voice in voices:
            if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                break
        self.engine.setProperty('rate', 160)
        self.engine.setProperty('volume', 1.0)

    def speak_alert(self, detections):
        try:
            now = time()
            if now - self.last_spoken < self.speak_cooldown:
                return

            new_status = "unsafe" if detections else "safe"

            if new_status != self.last_detection_status:
                self.last_detection_status = new_status
                if new_status == "unsafe":
                    message = " and ".join(set(detections))
                    self.engine.say(f"{message} detected here.")
                    self.update_status_button("Unsafe")
                else:
                    self.engine.say("Safe road ahead.")
                    self.update_status_button("Safe")
                self.engine.runAndWait()
                self.last_spoken = now
        except Exception as e:
            print("TTS error:", e)

    # --- Status Button ---
    def update_status_button(self, status):
        if status == "Safe":
            self.status_button.text = "\u2705 Safe Journey"
            self.status_button.color = (0, 1, 0, 1)
        elif status == "Unsafe":
            self.status_button.text = "\u26A0\ufe0f Unsafe Road Ahead"
            self.status_button.color = (1, 0, 0, 1)
        else:
            self.status_button.text = "Status: Unknown"
            self.status_button.color = (0.5, 0.5, 0.5, 1)

    # --- File Chooser ---
    def open_filechooser(self, instance):
        layout = BoxLayout(orientation='vertical')
        chooser = FileChooserIconView(size_hint=(1, 0.9))
        chooser.bind(on_selection=self.load_selected_image)
        layout.add_widget(chooser)
        popup = Popup(title='Select Image', content=layout, size_hint=(0.9, 0.9))
        chooser.popup = popup
        popup.open()

    def load_selected_image(self, chooser, selection):
        if selection:
            path = selection[0]
            img = cv2.imread(path)
            if img is not None:
                annotated, detections = self.detect_objects(img)
                self.image_view.texture = image_to_texture(annotated)
                self.speak_alert(detections)
            chooser.popup.dismiss()

    # --- Camera / Video ---
    def capture_camera(self, instance):
        cap = cv2.VideoCapture(0)
        for _ in range(3):
            ret, frame = cap.read()
        if ret:
            annotated, detections = self.detect_objects(frame)
            self.image_view.texture = image_to_texture(annotated)
            self.speak_alert(detections)
        cap.release()

    def start_video_stream(self, instance):
        self.live_only = False
        self.release_camera()
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update_video, 1.0 / 10.0)

    def start_live_stream(self, instance):
        self.live_only = True
        self.release_camera()
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update_live_video, 1.0 / 10.0)

    def update_video(self, dt):
        if self.capture:
            ret, frame = self.capture.read()
            if ret:
                annotated, detections = self.detect_objects(frame)
                self.image_view.texture = image_to_texture(annotated)
                self.speak_alert(detections)

    def update_live_video(self, dt):
        if self.capture:
            ret, frame = self.capture.read()
            if ret:
                annotated, detections = self.detect_objects(frame)
                self.image_view.texture = image_to_texture(annotated)
                self.speak_alert(detections)

    def release_camera(self):
        Clock.unschedule(self.update_video)
        Clock.unschedule(self.update_live_video)
        if self.capture and self.capture.isOpened():
            self.capture.release()
        self.capture = None

    # --- Accelerometer ---
    def start_accelerometer(self):
        try:
            accelerometer.enable()
            Clock.schedule_interval(self.read_accelerometer, 1.0 / 5.0)
        except NotImplementedError:
            print("Accelerometer not supported.")

    def read_accelerometer(self, dt):
        try:
            val = accelerometer.acceleration
            if val != (None, None, None):
                x, y, z = val
                print(f"Accelerometer: x={x:.2f}, y={y:.2f}, z={z:.2f}")
                if abs(z) > 15:
                    print("⚠️ Sudden vertical jolt detected!")
                    self.say("Possible pothole impact detected.")
                    self.update_status_button("Unsafe")
        except Exception as e:
            print(f"Accelerometer error: {e}")

    def stop_accelerometer(self):
        try:
            accelerometer.disable()
            Clock.unschedule(self.read_accelerometer)
        except:
            pass

    # --- GPS ---
    def start_gps(self):
        try:
            gps.configure(on_location=self.on_location, on_status=self.on_status)
            gps.start()
        except NotImplementedError:
            print("GPS not available on this platform")

    def on_location(self, **kwargs):
        lat = kwargs.get('lat')
        lon = kwargs.get('lon')
        print(f"Location: {lat}, {lon}")
        now = time()
        if now - self.last_spoken > self.speak_cooldown:
            if self.is_safe(lat, lon):
                self.say("Your safe journey starts now.")
                self.update_status_button("Safe")
            else:
                self.say("Warning: Unsafe road ahead.")
                self.update_status_button("Unsafe")
            self.last_spoken = now

    def is_safe(self, lat, lon):
        unsafe_coords = [(12.9716, 77.5946)]
        for ulat, ulon in unsafe_coords:
            if abs(ulat - lat) < 0.01 and abs(ulon - lon) < 0.01:
                return False
        return True

    def on_status(self, stype, status):
        print(f"GPS Status: {stype} - {status}")

    # --- Voice Control ---
    def say(self, message):
        self.engine.say(message)
        self.engine.runAndWait()

    def listen_voice_commands(self):
        recognizer = sr.Recognizer()
        mic = sr.Microphone()

        with mic as source:
            recognizer.adjust_for_ambient_noise(source)

        while True:
            with mic as source:
                print("🎙 Listening for voice command...")
                try:
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                    command = recognizer.recognize_google(audio).lower()
                    print("🗣 Heard command:", command)

                    if "start detection" in command or "start live" in command:
                        self.start_live_stream(None)
                    elif "stop detection" in command or "stop live" in command:
                        self.release_camera()
                    elif "upload image" in command:
                        self.open_filechooser(None)
                    elif "start video" in command:
                        self.start_video_stream(None)
                except sr.UnknownValueError:
                    print("Could not understand the audio.")
                except sr.WaitTimeoutError:
                    print("Listening timeout.")
                except sr.RequestError as e:
                    print(f"Speech recognition error: {e}")
                except Exception as e:
                    print(f"Voice command error: {e}")


# --- App ---
class DetectionGPSApp(App):
    def build(self):
        return MainLayout()


if __name__ == '__main__':
    DetectionGPSApp().run()
