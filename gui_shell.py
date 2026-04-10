"""
AudioTexture GUI with Real-Time Mel-Spectrogram Rendering
This GUI receives dropped audio files, converts them to Mel-spectrograms with Librosa,
and renders the result live in the canvas.
"""

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import threading
import os
import shutil
import warnings
import time
from pathlib import Path
from io import BytesIO

import numpy as np
import librosa
import librosa.display
try:
    import tensorflow as tf
    from tensorflow import keras
except Exception:
    tf = None
    keras = None
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

try:
    from audioread.exceptions import NoBackendError
except Exception:
    NoBackendError = RuntimeError

# Signal processing constants aligned with your workplan
SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
DURATION = 30.0
CHUNK_DURATION = 5.0

# Try to import TkinterDnD2
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    HAS_DND = True
except ImportError:
    print("Warning: tkinterdnd2 not installed. Drag-and-drop will be limited.")
    HAS_DND = False
    TkinterDnD = None
    DND_FILES = None


class AudioTextureGUI:
    """Main GUI class for AudioTexture application."""
    
    def __init__(self, root):
        """Initialize the AudioTexture GUI."""
        self.root = root
        self.root.title("AudioTexture - Music Genre Classification")
        self.root.geometry("1000x800")
        self.root.resizable(True, True)
        
        # Current state
        self.current_file = None
        self.current_genre = None
        self.current_confidence = None
        self.current_spec_image = None
        self.current_mel_image = None
        self.current_mel_shape = None
        self.processing_active = False
        self.processing_started_at = None
        self.telemetry_after_id = None

        # Prediction engine (stub today, swappable with trained model later)
        self.predictor = GenrePredictor()
        
        # Configure style
        self.setup_styles()
        
        # Build the layout
        self.build_ui()
        
    
    def setup_styles(self):
        """Configure ttk styles for modern appearance."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        bg_color = "#f0f0f0"
        accent_color = "#2196F3"
        text_color = "#333333"
        
        style.configure('TFrame', background=bg_color)
        style.configure('TLabel', background=bg_color, foreground=text_color)
        style.configure('Header.TLabel', font=('Segoe UI', 16, 'bold'), 
                       background=bg_color, foreground=accent_color)
        style.configure('Subheader.TLabel', font=('Segoe UI', 12, 'bold'),
                       background=bg_color, foreground=text_color)
        style.configure('Info.TLabel', font=('Segoe UI', 10),
                       background=bg_color, foreground=text_color)
        
        self.root.configure(bg=bg_color)
    
    
    def build_ui(self):
        """Build the main UI layout."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header
        header_label = ttk.Label(main_frame, text="AudioTexture", 
                                 style='Header.TLabel')
        header_label.pack(pady=(0, 10))
        
        subtitle_label = ttk.Label(main_frame, 
                                   text="AI-Powered Music Genre Classification",
                                   style='Info.TLabel')
        subtitle_label.pack(pady=(0, 30))
        
        # Main content area - split into two columns
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left side - Drop zone and file info
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.build_drop_zone(left_frame)
        self.build_file_info_panel(left_frame)
        
        # Right side - Results and spectrogram
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        self.build_results_panel(right_frame)
    
    
    def build_drop_zone(self, parent):
        """Build the drag-and-drop zone."""
        drop_frame = ttk.LabelFrame(parent, text="Drop Zone", padding=20)
        drop_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Create a canvas for the dashed border effect
        self.drop_canvas = tk.Canvas(drop_frame, bg="white", highlightthickness=2,
                                     highlightbackground="#CCCCCC", height=200)
        self.drop_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Dashed border rectangle
        self.drop_canvas.create_rectangle(5, 5, 395, 195, outline="#2196F3",
                                          width=2, dash=(5, 5))
        
        # Instructions text
        self.drop_text = self.drop_canvas.create_text(
            200, 100,
            text="Drag and drop MP3 or WAV file here",
            font=('Segoe UI', 14, 'bold'),
            fill="#999999"
        )
        
        # Bind drag-and-drop events
        if HAS_DND and DND_FILES is not None:
            drop_register = getattr(self.drop_canvas, 'drop_target_register', None)
            dnd_bind = getattr(self.drop_canvas, 'dnd_bind', None)
            if callable(drop_register) and callable(dnd_bind):
                try:
                    drop_register(DND_FILES)
                    dnd_bind('<<Drop>>', self.on_file_drop)
                    dnd_bind('<<DragEnter>>', self.on_drag_enter)
                    dnd_bind('<<DragLeave>>', self.on_drag_leave)
                except tk.TclError:
                    # tkdnd command hooks are unavailable in this Tk root; keep browse fallback.
                    pass

        # Fallback click-to-browse support if drag-and-drop is unavailable
        self.drop_canvas.bind('<Button-1>', self.on_browse_click)
    
    
    def on_drag_enter(self, event):
        """Handle drag enter event."""
        self.drop_canvas.configure(highlightbackground="#2196F3")
        self.drop_canvas.itemconfig(self.drop_text, fill="#2196F3")
    
    
    def on_drag_leave(self, event):
        """Handle drag leave event."""
        self.drop_canvas.configure(highlightbackground="#CCCCCC")
        if not self.current_file:
            self.drop_canvas.itemconfig(self.drop_text, fill="#999999")
    
    
    def on_file_drop(self, event):
        """Handle file drop event."""
        # Extract file path from event data
        file_path = event.data
        
        # Clean up file path (remove extra braces on Windows)
        if file_path.startswith('{') and file_path.endswith('}'):
            file_path = file_path[1:-1]
        
        # Validate file extension
        valid_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg']
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext not in valid_extensions:
            self.show_error(f"Invalid file type: {file_ext}\nSupported: {', '.join(valid_extensions)}")
            return
        
        # Set current file and update UI
        self.current_file = file_path
        self.update_file_display(file_path)
        
        # Start real audio processing pipeline
        self.start_processing(file_path)


    def on_browse_click(self, _event):
        """Fallback file picker for environments without drag-and-drop support."""
        file_path = filedialog.askopenfilename(
            title="Select an audio file",
            filetypes=[
                ("Audio Files", "*.mp3 *.wav *.m4a *.flac *.ogg"),
                ("All Files", "*.*")
            ]
        )
        if not file_path:
            return

        self.current_file = file_path
        self.update_file_display(file_path)
        self.start_processing(file_path)
    
    
    def update_file_display(self, file_path):
        """Update the display to show the selected file."""
        file_name = Path(file_path).name
        self.drop_canvas.itemconfig(self.drop_text, 
                                   text=f"✓ File loaded:\n{file_name}",
                                   fill="#4CAF50")
        self.file_info_content.config(state=tk.NORMAL)
        self.file_info_content.delete('1.0', tk.END)
        self.file_info_content.insert('1.0', f"File: {file_name}\nPath: {file_path}")
        self.file_info_content.config(state=tk.DISABLED)
    
    
    def build_file_info_panel(self, parent):
        """Build the file information panel."""
        info_frame = ttk.LabelFrame(parent, text="File Information", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.file_info_content = tk.Text(info_frame, height=5, width=40,
                                         state=tk.DISABLED, wrap=tk.WORD,
                                         font=('Courier New', 10))
        self.file_info_content.pack(fill=tk.BOTH, expand=True)
        
        # Add placeholder text
        self.file_info_content.config(state=tk.NORMAL)
        self.file_info_content.insert('1.0', "Awaiting file...\n\nDrop an audio file in the zone above.")
        self.file_info_content.config(state=tk.DISABLED)
    
    
    def build_results_panel(self, parent):
        """Build the results panel (genre prediction)."""
        # Results header
        results_label = ttk.Label(parent, text="Classification Results", 
                                  style='Subheader.TLabel')
        results_label.pack(pady=(0, 15))
        
        # Results frame with border
        results_frame = ttk.LabelFrame(parent, text="Genre Prediction", padding=20)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Genre display (large)
        self.genre_label = ttk.Label(results_frame, 
                                     text="Awaiting classification...",
                                     font=('Segoe UI', 24, 'bold'),
                                     foreground="#2196F3")
        self.genre_label.pack(pady=20)
        
        # Confidence display
        confidence_frame = ttk.Frame(results_frame)
        confidence_frame.pack(fill=tk.X, pady=15)
        
        ttk.Label(confidence_frame, text="Confidence:", 
                 style='Info.TLabel').pack(side=tk.LEFT, padx=(0, 10))
        
        self.confidence_label = ttk.Label(confidence_frame, 
                                         text="0%",
                                         font=('Segoe UI', 14, 'bold'),
                                         foreground="#4CAF50")
        self.confidence_label.pack(side=tk.LEFT)
        
        # Progress bar for processing
        ttk.Label(results_frame, text="Processing Status:", 
                 style='Info.TLabel').pack(pady=(15, 5))
        
        self.progress_bar = ttk.Progressbar(results_frame, length=300, 
                                           mode='indeterminate')
        self.progress_bar.pack(fill=tk.X, pady=(0, 15))
        
        # Status message
        self.status_label = ttk.Label(results_frame, text="Ready",
                                      style='Info.TLabel', foreground="#666666")
        self.status_label.pack()

        self.metrics_label = ttk.Label(
            results_frame,
            text="",
            style='Info.TLabel',
            foreground="#666666"
        )
        self.metrics_label.pack(pady=(4, 0))
        
        # Spectrogram canvas placeholder
        ttk.Label(results_frame, text="Mel-Spectrogram Visualization:", 
                 style='Info.TLabel').pack(pady=(20, 10))
        
        self.spec_canvas = tk.Canvas(results_frame, bg="#f5f5f5", 
                                     highlightbackground="#CCCCCC",
                                     height=150)
        self.spec_canvas.pack(fill=tk.BOTH, expand=True)

        # Telemetry overlay drawn on top of the spectrogram canvas.
        self.telemetry_overlay_bg = self.spec_canvas.create_rectangle(
            8, 8, 310, 44,
            fill="#000000",
            outline="",
            stipple="gray50",
            state='hidden'
        )
        self.telemetry_overlay_text = self.spec_canvas.create_text(
            16, 26,
            text="",
            fill="#ffffff",
            anchor='w',
            font=('Segoe UI', 9, 'bold'),
            state='hidden'
        )

        self.save_button = ttk.Button(
            results_frame,
            text="Save Spectrogram",
            command=self.save_current_spectrogram,
            state=tk.DISABLED
        )
        self.save_button.pack(pady=(10, 0), anchor='e')
        
        # Draw placeholder
        self.placeholder_text_id = self.spec_canvas.create_text(
            220,
            75,
            text="[Spectrogram will display here]",
            fill="#CCCCCC",
            font=('Segoe UI', 12)
        )
    
    
    def start_processing(self, file_path):
        """Start processing the file in a background thread."""
        # Prevent launching multiple long-running FFT jobs at once.
        if self.processing_active:
            self.status_label.config(text="Processing already in progress...", foreground="#FF9800")
            return

        self.processing_active = True
        self.processing_started_at = time.perf_counter()
        self.progress_bar.start()
        self.status_label.config(text="Generating Mel-spectrogram...", foreground="#FF9800")
        self.metrics_label.config(text="Latency: calculating... | Mel shape: pending")
        self.genre_label.config(text="Analyzing...")
        self.confidence_label.config(text="--")
        self.save_button.config(state=tk.DISABLED)
        self._set_canvas_telemetry("Processing... 0 ms | Mel: pending")
        self._start_telemetry_loop()
        
        # Run processing in background thread
        thread = threading.Thread(target=self.process_file, args=(file_path,))
        thread.daemon = True
        thread.start()
    
    
    def process_file(self, file_path):
        """Convert audio file to Mel-spectrogram and update UI."""
        try:
            t0 = time.perf_counter()
            mel_image, mel_db = audio_to_mel_image(file_path)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            predicted_genre, confidence = self.predictor.predict_from_audio(
                file_path,
                mel_db=mel_db,
                mel_image=mel_image
            )

            self.root.after(
                0,
                self.update_results,
                predicted_genre,
                confidence,
                mel_image,
                mel_db.shape,
                elapsed_ms
            )
        except Exception as e:
            self.root.after(0, self.show_error, f"Processing failed: {e}")
    
    
    def update_results(self, genre, confidence, mel_image, mel_shape, elapsed_ms):
        """Update the results display."""
        self.processing_active = False
        self._stop_telemetry_loop()
        self.progress_bar.stop()
        self.current_genre = genre
        self.current_confidence = confidence
        self.current_mel_image = mel_image
        self.current_mel_shape = mel_shape
        
        self.genre_label.config(text=genre)
        self.confidence_label.config(text=f"{confidence}%")
        self.status_label.config(text="Mel-spectrogram generated and displayed", foreground="#4CAF50")
        self.metrics_label.config(text=f"Processing time: {elapsed_ms:.0f} ms | Mel shape: {mel_shape}")
        self.render_spectrogram_on_canvas(mel_image)
        self._hide_canvas_telemetry()
        self.save_button.config(state=tk.NORMAL)


    def render_spectrogram_on_canvas(self, image):
        """Render a PIL image into the spectrogram canvas."""
        self.spec_canvas.update_idletasks()
        canvas_w = self.spec_canvas.winfo_width() or 440
        canvas_h = self.spec_canvas.winfo_height() or 150

        # Fit image to canvas while preserving aspect ratio.
        img_copy = image.copy()
        img_copy.thumbnail((canvas_w - 4, canvas_h - 4), Image.Resampling.LANCZOS)

        self.current_spec_image = ImageTk.PhotoImage(img_copy)
        self.spec_canvas.delete('spec_image')
        placeholder_id = getattr(self, 'placeholder_text_id', None)
        if placeholder_id is not None:
            self.spec_canvas.delete(placeholder_id)
            self.placeholder_text_id = None
        self.spec_canvas.create_image(
            canvas_w // 2,
            canvas_h // 2,
            image=self.current_spec_image,
            tags='spec_image'
        )
        self.spec_canvas.tag_lower('spec_image')
        self.spec_canvas.tag_raise(self.telemetry_overlay_bg)
        self.spec_canvas.tag_raise(self.telemetry_overlay_text)
    
    
    def show_error(self, message):
        """Display an error message."""
        self.processing_active = False
        self._stop_telemetry_loop()
        self.progress_bar.stop()
        self.status_label.config(text=f"Error: {message}", foreground="#F44336")
        self.metrics_label.config(text="")
        self._hide_canvas_telemetry()
        self.save_button.config(state=tk.DISABLED)


    def _start_telemetry_loop(self):
        """Start periodic UI-side telemetry updates while worker thread runs."""
        self._stop_telemetry_loop()
        self.telemetry_after_id = self.root.after(120, self._update_live_telemetry)


    def _stop_telemetry_loop(self):
        """Cancel periodic telemetry updates."""
        if self.telemetry_after_id is not None:
            self.root.after_cancel(self.telemetry_after_id)
            self.telemetry_after_id = None


    def _update_live_telemetry(self):
        """Update latency text while processing is still active."""
        if not self.processing_active or self.processing_started_at is None:
            return

        elapsed_ms = (time.perf_counter() - self.processing_started_at) * 1000.0
        text = f"Latency: {elapsed_ms:.0f} ms | Mel shape: pending"
        self.metrics_label.config(text=text)
        self._set_canvas_telemetry(f"Processing... {elapsed_ms:.0f} ms | Mel: pending")
        self.telemetry_after_id = self.root.after(120, self._update_live_telemetry)


    def _set_canvas_telemetry(self, text):
        """Show/update telemetry text overlay in the spectrogram canvas."""
        self.spec_canvas.itemconfigure(self.telemetry_overlay_text, text=text)
        self.spec_canvas.itemconfigure(self.telemetry_overlay_bg, state='normal')
        self.spec_canvas.itemconfigure(self.telemetry_overlay_text, state='normal')
        self.spec_canvas.tag_raise(self.telemetry_overlay_bg)
        self.spec_canvas.tag_raise(self.telemetry_overlay_text)


    def _hide_canvas_telemetry(self):
        """Hide telemetry overlay from the spectrogram canvas."""
        self.spec_canvas.itemconfigure(self.telemetry_overlay_bg, state='hidden')
        self.spec_canvas.itemconfigure(self.telemetry_overlay_text, state='hidden')


    def save_current_spectrogram(self):
        """Save the currently rendered Mel-spectrogram to disk."""
        if self.current_mel_image is None:
            self.show_error("No spectrogram available to save")
            return

        base_name = "spectrogram"
        if self.current_file:
            base_name = Path(self.current_file).stem + "_mel"

        save_path = filedialog.asksaveasfilename(
            title="Save spectrogram image",
            defaultextension=".png",
            initialfile=f"{base_name}.png",
            filetypes=[("PNG Image", "*.png")]
        )
        if not save_path:
            return

        try:
            self.current_mel_image.save(save_path, format="PNG")
            self.status_label.config(text=f"Saved spectrogram: {Path(save_path).name}", foreground="#4CAF50")
        except Exception as exc:
            self.show_error(f"Save failed: {exc}")


class GenrePredictor:
    """
    Model hook for genre inference.
    This currently uses a deterministic mel-feature stub and is ready to be
    replaced with a trained Keras model in Checkpoint 3.
    """

    def __init__(self):
        self.labels = [
            'Electronic', 'Experimental', 'Folk', 'Hip-Hop',
            'Instrumental', 'International', 'Pop', 'Rock'
        ]
        self.model = None
        self.model_name = None
        self.use_nn = False
        self.input_mode = 'engineered_3ch'
        self.use_chunk_voting = True
        self._load_trained_models()


    def _configure_inference_profile(self):
        """Select preprocessing/chunking profile based on loaded checkpoint family."""
        name = (self.model_name or '').lower()

        # Models trained on chunked + engineered pipeline.
        if any(k in name for k in ('chunk_retrain', 'adv_resnet', 'resnet50_focal', 'adv_effnet', 'focal')):
            self.input_mode = 'engineered_3ch'
            self.use_chunk_voting = True
            return

        # Earlier single-image models were trained on RGB spectrogram images.
        self.input_mode = 'raw_rgb'
        self.use_chunk_voting = False


    def _load_trained_models(self):
        """Load a single best available Keras model for inference."""
        if tf is None:
            print("TensorFlow not available; using deterministic fallback predictor.")
            return

        project_root = Path(__file__).resolve().parent
        candidate_dirs = [
            project_root / 'Model data' / 'model_checkpoints',
            project_root / 'model_checkpoints',
        ]

        ckpt_dir = None
        for path in candidate_dirs:
            if path.exists():
                ckpt_dir = path
                break

        if ckpt_dir is None:
            print("No checkpoint directory found; using deterministic fallback predictor.")
            return

        # Priority order: best production candidates first.
        preferred = [
            'final_chunk_retrained_resnet50.keras',
            'final_adv_resnet50.keras',
            'chunk_retrain_stage3_best.keras',
            'chunk_retrain_stage2_best.keras',
            'adv_resnet50_stage3_best.keras',
            'final_adv_effnetv2b0.keras',
            'adv_effnetv2b0_stage3_best.keras',
            'finetuned_best.keras',
            'hybrid_best.keras',
            'refined_best.keras',
            'baseline_best.keras',
        ]

        # Define ApplicationPreprocess for export-safe model deserialization
        @keras.utils.register_keras_serializable(package='AudioTexture')
        class ApplicationPreprocess(tf.keras.layers.Layer):
            def __init__(self, mode, **kwargs):
                super().__init__(**kwargs)
                self.mode = mode

            def call(self, inputs):
                if self.mode == 'resnet50':
                    return tf.keras.applications.resnet.preprocess_input(inputs)
                if self.mode == 'efficientnetv2b0':
                    return tf.keras.applications.efficientnet_v2.preprocess_input(inputs)
                raise ValueError(f'Unknown preprocess mode: {self.mode}')

            def get_config(self):
                config = super().get_config()
                config.update({'mode': self.mode})
                return config

        custom_objects = {
            'preprocess_input': tf.keras.applications.resnet.preprocess_input,
            'ApplicationPreprocess': ApplicationPreprocess,
        }

        for name in preferred:
            ckpt_path = ckpt_dir / name
            if not ckpt_path.exists():
                continue
            try:
                self.model = tf.keras.models.load_model(
                    str(ckpt_path),
                    compile=False,
                    safe_mode=False,
                    custom_objects=custom_objects,
                )
                self.model_name = name
                self.use_nn = True
                self._configure_inference_profile()
                print(f"Loaded best model: {name}")
                print(f"Inference profile -> mode: {self.input_mode}, chunk_voting: {self.use_chunk_voting}")
                return
            except Exception as exc:
                print(f"Skipping model {name}: {exc}")

        print("No loadable checkpoints found; using deterministic fallback predictor.")


    def _prepare_engineered_tensor(self, mel_image):
        """Build the same 3-channel engineered input used by advanced notebook training."""
        if tf is None:
            raise RuntimeError("TensorFlow is unavailable for neural inference.")
        tfm = tf

        # Keep preprocessing aligned with the notebook's advanced path.
        img = np.asarray(mel_image.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        x = tfm.convert_to_tensor(img)
        gray = tfm.image.rgb_to_grayscale(x)         # [H, W, 1]
        gray_b = tfm.expand_dims(gray, axis=0)       # [1, H, W, 1]
        sobel = tfm.image.sobel_edges(gray_b)        # [1, H, W, 1, 2]
        sobel = tfm.squeeze(sobel, axis=0)           # [H, W, 1, 2]
        gx = sobel[..., 0]
        gy = sobel[..., 1]
        gx = tfm.clip_by_value((gx + 1.0) * 0.5, 0.0, 1.0)
        gy = tfm.clip_by_value((gy + 1.0) * 0.5, 0.0, 1.0)
        engineered = tfm.concat([gray, gx, gy], axis=-1)
        return tfm.expand_dims(engineered, axis=0)   # [1, H, W, 3]


    def _prepare_raw_rgb_tensor(self, mel_image):
        """Prepare plain RGB tensor for models trained on raw spectrogram images."""
        if tf is None:
            raise RuntimeError("TensorFlow is unavailable for neural inference.")

        # Raw-RGB checkpoints in this project include internal preprocessing
        # (Rescaling or preprocess_input), so keep pixel scale at 0..255.
        img = np.asarray(
            mel_image.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
        return tf.expand_dims(tf.convert_to_tensor(img), axis=0)


    def _predict_with_models(self, mel_image):
        """Run prediction with the loaded neural model."""
        if self.model is None:
            raise RuntimeError("No neural model is loaded.")

        if self.input_mode == 'raw_rgb':
            x = self._prepare_raw_rgb_tensor(mel_image)
        else:
            x = self._prepare_engineered_tensor(mel_image)
        probs = self.model(x, training=False).numpy()[0].astype(np.float32)
        denom = float(np.sum(probs))
        if denom <= 0.0:
            raise RuntimeError("Model returned invalid probabilities.")
        probs = probs / denom
        idx = int(np.argmax(probs))
        confidence = int(round(float(probs[idx]) * 100.0))
        return self.labels[idx], max(1, min(confidence, 99))


    def _load_audio_for_chunks(self, file_path):
        """Load and normalize audio to a fixed 30-second window for chunk voting."""
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='PySoundFile failed. Trying audioread instead.')
            warnings.filterwarnings('ignore', message='librosa.core.audio.__audioread_load')
            audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION, mono=True)

        target_len = int(SAMPLE_RATE * DURATION)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)), mode='constant')
        else:
            audio = audio[:target_len]
        return audio


    def _predict_from_audio_chunks(self, file_path):
        """Predict with 5-second chunk voting using probability averaging."""
        if self.model is None:
            raise RuntimeError("No neural model is loaded.")
        model = self.model

        audio = self._load_audio_for_chunks(file_path)
        chunk_len = int(SAMPLE_RATE * CHUNK_DURATION)
        num_chunks = int(DURATION / CHUNK_DURATION)

        probs_acc = np.zeros((len(self.labels),), dtype=np.float32)
        used = 0

        for i in range(num_chunks):
            start = i * chunk_len
            end = start + chunk_len
            chunk_audio = audio[start:end]
            if len(chunk_audio) < chunk_len:
                chunk_audio = np.pad(chunk_audio, (0, chunk_len - len(chunk_audio)), mode='constant')

            mel_spec = librosa.feature.melspectrogram(
                y=chunk_audio,
                sr=SAMPLE_RATE,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_mels=N_MELS,
                fmax=8000
            )
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)

            chunk_image = mel_db_to_model_image(mel_db)
            if self.input_mode == 'raw_rgb':
                x = self._prepare_raw_rgb_tensor(chunk_image)
            else:
                x = self._prepare_engineered_tensor(chunk_image)
            pred = model(x, training=False).numpy()[0].astype(np.float32)

            probs_acc += pred
            used += 1

        if used == 0:
            raise RuntimeError("No chunks available for prediction.")

        probs = probs_acc / float(used)
        denom = float(np.sum(probs))
        if denom <= 0.0:
            raise RuntimeError("Chunk voting produced invalid probabilities.")

        probs = probs / denom
        idx = int(np.argmax(probs))
        confidence = int(round(float(probs[idx]) * 100.0))
        return self.labels[idx], max(1, min(confidence, 99))


    def _predict_stub(self, mel_db):
        """Fallback deterministic predictor if no trained model is available."""
        # Feature vector derived from spectral texture
        mean_energy = float(np.mean(mel_db))
        spread = float(np.std(mel_db))
        high_band = float(np.mean(mel_db[-16:, :]))
        low_band = float(np.mean(mel_db[:16, :]))
        rhythm_proxy = float(np.std(np.diff(mel_db, axis=1)))

        features = np.array([mean_energy, spread, high_band, low_band, rhythm_proxy], dtype=np.float32)

        # Fixed random-like projection for deterministic pseudo-probabilities
        W = np.array([
            [0.4, -0.2, 0.3, -0.1, 0.2],
            [0.2, 0.3, -0.1, 0.4, -0.2],
            [-0.1, 0.5, -0.3, 0.2, 0.1],
            [0.3, -0.4, 0.2, 0.1, 0.2],
            [-0.3, 0.2, 0.4, -0.2, 0.3],
            [0.1, -0.1, 0.2, 0.4, 0.3],
            [0.2, 0.1, -0.2, 0.3, -0.1],
            [0.3, 0.2, 0.1, -0.3, 0.2],
        ], dtype=np.float32)
        b = np.array([0.3, 0.1, 0.0, 0.2, -0.1, 0.0, 0.15, 0.05], dtype=np.float32)

        logits = W @ features + b
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / np.sum(probs)

        idx = int(np.argmax(probs))
        confidence = int(round(float(probs[idx]) * 100.0))
        return self.labels[idx], max(1, min(confidence, 99))

    def predict_from_mel(self, mel_db, mel_image=None):
        """
        Return (genre, confidence_percent) using trained model(s) when available.
        """
        if self.use_nn:
            try:
                # Always infer from training-style rendering to avoid UI-style drift.
                model_image = mel_db_to_model_image(mel_db)
                return self._predict_with_models(model_image)
            except Exception as exc:
                print(f"Neural inference failed, falling back to stub predictor: {exc}")

        return self._predict_stub(mel_db)


    def predict_from_audio(self, file_path, mel_db=None, mel_image=None):
        """Primary inference path: chunk-voting neural prediction with safe fallback."""
        if self.use_nn:
            if self.use_chunk_voting:
                try:
                    return self._predict_from_audio_chunks(file_path)
                except Exception as exc:
                    print(f"Chunk voting failed; attempting single-image fallback: {exc}")
                    if mel_db is not None:
                        try:
                            model_image = mel_db_to_model_image(mel_db)
                            return self._predict_with_models(model_image)
                        except Exception as exc2:
                            print(f"Single-image neural inference failed: {exc2}")
            else:
                if mel_db is not None:
                    try:
                        model_image = mel_db_to_model_image(mel_db)
                        return self._predict_with_models(model_image)
                    except Exception as exc:
                        print(f"Single-image neural inference failed; falling back to stub: {exc}")

        if mel_db is not None:
            return self._predict_stub(mel_db)

        # Last-resort fallback if mel_db wasn't provided.
        try:
            _, mel_db_local = audio_to_mel_image(file_path)
            return self._predict_stub(mel_db_local)
        except Exception as exc:
            raise RuntimeError(f"Fallback prediction failed: {exc}") from exc


def audio_to_mel_image(file_path):
    """
    Convert an audio file into a Mel-spectrogram PIL image.
    """
    _ensure_ffmpeg_backend()

    # Load and normalize duration
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='PySoundFile failed. Trying audioread instead.')
            warnings.filterwarnings('ignore', message='librosa.core.audio.__audioread_load')
            audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION, mono=True)
    except NoBackendError as exc:
        raise RuntimeError(
            "Could not decode this audio format. Install ffmpeg or convert the file to WAV/MP3."
        ) from exc
    except Exception as exc:
        # Provide a clearer message for unsupported/corrupt files.
        raise RuntimeError(f"Audio load failed: {exc}") from exc

    target_len = int(SAMPLE_RATE * DURATION)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)), mode='constant')
    else:
        audio = audio[:target_len]

    # Generate Mel-spectrogram and convert to dB scale
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmax=8000
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Render to in-memory PNG and return PIL image
    fig, ax = plt.subplots(figsize=(4.4, 1.5), dpi=100)
    librosa.display.specshow(
        mel_db,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis='time',
        y_axis='mel',
        cmap='magma',
        ax=ax
    )
    ax.set_title('Mel-Spectrogram', fontsize=9)
    ax.tick_params(axis='both', labelsize=7)
    plt.tight_layout(pad=0.2)

    buf = BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGB'), mel_db


def mel_db_to_model_image(mel_db, img_size=(224, 224)):
    """Render mel dB array into the same image style used during batch training."""
    dpi = 100
    figsize = (img_size[0] / dpi, img_size[1] / dpi)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    librosa.display.specshow(
        mel_db,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis=None,
        y_axis=None,
        ax=ax,
        cmap='viridis'
    )
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def _ensure_ffmpeg_backend():
    """
    Make a bundled ffmpeg binary discoverable for audioread when available.
    """
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_path = Path(ffmpeg_exe)
        ffmpeg_dir = str(ffmpeg_path.parent)

        # audioread searches for "ffmpeg" command name specifically.
        ffmpeg_alias = ffmpeg_path.parent / 'ffmpeg.exe'
        if not ffmpeg_alias.exists():
            try:
                shutil.copyfile(ffmpeg_path, ffmpeg_alias)
            except Exception:
                pass

        current_path = os.environ.get('PATH', '')
        if ffmpeg_dir and ffmpeg_dir not in current_path:
            os.environ['PATH'] = ffmpeg_dir + os.pathsep + current_path
    except Exception:
        # If unavailable, librosa will still try native backends.
        pass


def main():
    """Main entry point."""
    if HAS_DND and TkinterDnD is not None:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    app = AudioTextureGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
