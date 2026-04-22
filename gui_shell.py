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
import csv
from datetime import datetime
from pathlib import Path
from io import BytesIO

import numpy as np
import librosa
import librosa.display
try:
    import tensorflow as tf
except Exception:
    tf = None
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
CONFIDENCE_THRESHOLD = 50

UI_THEMES = {
    'Classic': {
        'bg': '#f0f0f0',
        'surface': '#ffffff',
        'canvas': '#f5f5f5',
        'accent': '#2196F3',
        'text': '#333333',
        'muted': '#666666',
        'success': '#4CAF50',
        'warning': '#FF9800',
        'error': '#F44336',
        'placeholder': '#CCCCCC',
    },
    'Modern': {
        'bg': '#eef5ff',
        'surface': '#ffffff',
        'canvas': '#edf2fb',
        'accent': '#0A66C2',
        'text': '#102A43',
        'muted': '#486581',
        'success': '#0F9D58',
        'warning': '#E67700',
        'error': '#C92A2A',
        'placeholder': '#9FB3C8',
    },
}

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
        self.root.geometry("1080x860")
        self.root.resizable(True, True)
        self.project_root = Path(__file__).resolve().parent
        self.ui_mode_var = tk.StringVar(value='Classic')
        self.current_palette = UI_THEMES['Classic']
        
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
        self._apply_theme_to_widgets()
        
    
    def setup_styles(self):
        """Configure ttk styles based on active UI mode."""
        style = ttk.Style()
        style.theme_use('clam')

        self.current_palette = UI_THEMES.get(self.ui_mode_var.get(), UI_THEMES['Classic'])
        p = self.current_palette

        style.configure('TFrame', background=p['bg'])
        style.configure('TLabel', background=p['bg'], foreground=p['text'])
        style.configure('Header.TLabel', font=('Segoe UI', 16, 'bold'), background=p['bg'], foreground=p['accent'])
        style.configure('Subheader.TLabel', font=('Segoe UI', 12, 'bold'), background=p['bg'], foreground=p['text'])
        style.configure('Info.TLabel', font=('Segoe UI', 10), background=p['bg'], foreground=p['text'])
        style.configure('TLabelframe', background=p['bg'], borderwidth=1)
        style.configure('TLabelframe.Label', background=p['bg'], foreground=p['text'])

        if self.ui_mode_var.get() == 'Modern':
            style.configure('Header.TLabel', font=('Segoe UI', 18, 'bold'))
            style.configure('Subheader.TLabel', font=('Segoe UI', 13, 'bold'))
            style.configure('Info.TLabel', font=('Segoe UI', 10))
            style.configure('TLabelframe', borderwidth=0)
        else:
            style.configure('Header.TLabel', font=('Segoe UI', 16, 'bold'))
            style.configure('Subheader.TLabel', font=('Segoe UI', 12, 'bold'))
            style.configure('TLabelframe', borderwidth=1)

        self.root.configure(bg=p['bg'])


    def on_ui_mode_change(self, *_args):
        """Switch between classic and modern UI palettes."""
        self.setup_styles()
        self._layout_by_mode()
        self._apply_theme_to_widgets()
        self._ensure_layout_visibility()


    def _apply_theme_to_widgets(self):
        """Apply current palette colors to non-ttk widgets and labels."""
        p = self.current_palette

        if hasattr(self, 'content_canvas'):
            self.content_canvas.configure(bg=p['bg'])

        if hasattr(self, 'drop_canvas'):
            self.drop_canvas.configure(bg=p['surface'], highlightbackground=p['placeholder'])
            if not self.current_file:
                self.drop_canvas.itemconfig(self.drop_text, fill=p['muted'])
            self._draw_drop_zone_chrome()

        if hasattr(self, 'file_info_content'):
            self.file_info_content.configure(bg=p['surface'], fg=p['text'])

        if hasattr(self, 'spec_canvas'):
            self.spec_canvas.configure(bg=p['canvas'], highlightbackground=p['placeholder'])
            self._draw_spec_canvas_chrome()

        if hasattr(self, 'placeholder_text_id') and self.placeholder_text_id is not None:
            self.spec_canvas.itemconfig(self.placeholder_text_id, fill=p['placeholder'])

        if hasattr(self, 'genre_label'):
            self.genre_label.configure(foreground=p['accent'])

        if hasattr(self, 'confidence_label'):
            self.confidence_label.configure(foreground=p['success'])

        if hasattr(self, 'status_label'):
            self.status_label.configure(foreground=p['muted'])
    
    
    def build_ui(self):
        """Build the main UI layout."""
        # Main container
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header
        header_label = ttk.Label(self.main_frame, text="AudioTexture", 
                                 style='Header.TLabel')
        header_label.pack(pady=(0, 10))
        
        subtitle_label = ttk.Label(self.main_frame, 
                                   text="AI-Powered Music Genre Classification",
                                   style='Info.TLabel')
        subtitle_label.pack(pady=(0, 30))

        controls_frame = ttk.Frame(self.main_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(controls_frame, text="UI Mode:", style='Info.TLabel').pack(side=tk.LEFT, padx=(0, 8))
        self.ui_mode_combo = ttk.Combobox(
            controls_frame,
            values=list(UI_THEMES.keys()),
            textvariable=self.ui_mode_var,
            state='readonly',
            width=10,
        )
        self.ui_mode_combo.pack(side=tk.LEFT, padx=(0, 12))
        self.ui_mode_combo.bind('<<ComboboxSelected>>', self.on_ui_mode_change)

        self.performance_button = ttk.Button(
            controls_frame,
            text="Performance Snapshot",
            command=self.show_performance_snapshot,
        )
        self.performance_button.pack(side=tk.LEFT, padx=(0, 8))

        self.self_check_button = ttk.Button(
            controls_frame,
            text="Run Demo Self-Check",
            command=self.run_demo_self_check,
        )
        self.self_check_button.pack(side=tk.LEFT)

        # Scrollable content area so Modern mode can stack safely on smaller windows.
        self.scroll_area = ttk.Frame(self.main_frame)
        self.scroll_area.pack(fill=tk.BOTH, expand=True)

        self.content_canvas = tk.Canvas(
            self.scroll_area,
            highlightthickness=0,
            bg=self.current_palette['bg'],
        )
        self.content_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.content_scrollbar = ttk.Scrollbar(
            self.scroll_area,
            orient=tk.VERTICAL,
            command=self.content_canvas.yview,
        )
        self.content_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.content_canvas.configure(yscrollcommand=self.content_scrollbar.set)

        self.content_frame = ttk.Frame(self.content_canvas)
        self.content_window_id = self.content_canvas.create_window((0, 0), window=self.content_frame, anchor='nw')

        self.content_frame.bind('<Configure>', self._sync_content_scrollregion)
        self.content_canvas.bind('<Configure>', self._sync_content_canvas_width)
        self.content_canvas.bind_all('<MouseWheel>', self._on_mousewheel)
        self.content_canvas.bind_all('<Button-4>', self._on_mousewheel)
        self.content_canvas.bind_all('<Button-5>', self._on_mousewheel)
        
        # Left side - Drop zone and file info
        self.left_frame = ttk.Frame(self.content_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.build_drop_zone(self.left_frame)
        self.build_file_info_panel(self.left_frame)
        
        # Right side - Results and spectrogram
        self.right_frame = ttk.Frame(self.content_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        self.build_results_panel(self.right_frame)
        self._layout_by_mode()
        self._ensure_layout_visibility()


    def _sync_content_scrollregion(self, _event=None):
        """Keep the scrollable region aligned with the embedded content frame."""
        if hasattr(self, 'content_canvas'):
            self.content_canvas.configure(scrollregion=self.content_canvas.bbox('all'))


    def _sync_content_canvas_width(self, event):
        """Match the embedded content frame width to the canvas width."""
        if hasattr(self, 'content_window_id'):
            self.content_canvas.itemconfigure(self.content_window_id, width=event.width)


    def _on_mousewheel(self, event):
        """Scroll the content canvas with the mouse wheel."""
        if not hasattr(self, 'content_canvas'):
            return

        if getattr(event, 'num', None) == 4:
            delta = -1
        elif getattr(event, 'num', None) == 5:
            delta = 1
        else:
            delta = int(-1 * (event.delta / 120)) if getattr(event, 'delta', 0) else 0

        if delta:
            self.content_canvas.yview_scroll(delta, 'units')


    def _layout_by_mode(self):
        """Reflow primary panes by selected mode to make layouts distinct."""
        if not hasattr(self, 'left_frame') or not hasattr(self, 'right_frame'):
            return

        self.left_frame.pack_forget()
        self.right_frame.pack_forget()

        if self.ui_mode_var.get() == 'Modern':
            # Modern mode: stacked sections, input tools first for immediate visibility.
            self.left_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 14))
            self.right_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.left_frame.configure(height=430)
            self.right_frame.configure(height=360)
            self.right_frame.pack_propagate(False)
            self.left_frame.pack_propagate(False)
        else:
            # Classic mode: two-column split.
            self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
            self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
            self.right_frame.pack_propagate(True)
            self.left_frame.pack_propagate(True)


    def _ensure_layout_visibility(self):
        """Make sure the active mode opens with enough room to show its main controls."""
        if self.ui_mode_var.get() == 'Modern':
            self.root.minsize(1040, 860)
            self.root.geometry('1040x860')
        else:
            self.root.minsize(1000, 800)
    
    
    def build_drop_zone(self, parent):
        """Build the drag-and-drop zone."""
        drop_frame = ttk.LabelFrame(parent, text="Drop Zone", padding=20)
        drop_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Create a canvas for the dashed border effect
        self.drop_canvas = tk.Canvas(drop_frame, bg="white", highlightthickness=2,
                                     highlightbackground="#CCCCCC", height=200)
        self.drop_canvas.pack(fill=tk.BOTH, expand=True)

        self.drop_canvas.bind('<Configure>', lambda _e: self._draw_drop_zone_chrome())
        
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


    def _create_rounded_rect(self, canvas, x1, y1, x2, y2, radius=16, **kwargs):
        """Draw a rounded rectangle on a Tk canvas using a smoothed polygon."""
        points = [
            x1 + radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1,
        ]
        return canvas.create_polygon(points, smooth=True, splinesteps=36, **kwargs)


    def _draw_drop_zone_chrome(self):
        """Draw mode-specific border treatment for the drop canvas."""
        if not hasattr(self, 'drop_canvas'):
            return

        canvas = self.drop_canvas
        canvas.delete('drop_chrome')
        w = max(canvas.winfo_width(), 40)
        h = max(canvas.winfo_height(), 40)
        p = self.current_palette

        if self.ui_mode_var.get() == 'Modern':
            self._create_rounded_rect(
                canvas,
                6,
                6,
                w - 6,
                h - 6,
                radius=22,
                outline=p['accent'],
                width=2,
                fill=p['surface'],
                tags='drop_chrome',
            )
        else:
            canvas.create_rectangle(
                5,
                5,
                w - 5,
                h - 5,
                outline=p['accent'],
                width=2,
                dash=(5, 5),
                tags='drop_chrome',
            )

        canvas.tag_lower('drop_chrome')
    
    
    def on_drag_enter(self, event):
        """Handle drag enter event."""
        self.drop_canvas.configure(highlightbackground=self.current_palette['accent'])
        self.drop_canvas.itemconfig(self.drop_text, fill=self.current_palette['accent'])
    
    
    def on_drag_leave(self, event):
        """Handle drag leave event."""
        self.drop_canvas.configure(highlightbackground=self.current_palette['placeholder'])
        if not self.current_file:
            self.drop_canvas.itemconfig(self.drop_text, fill=self.current_palette['muted'])
    
    
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

        self.batch_button = ttk.Button(
            info_frame,
            text="Batch Folder -> CSV",
            command=self.on_batch_infer_click,
        )
        self.batch_button.pack(pady=(8, 0), anchor='e')
    
    
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

        self.top3_label = ttk.Label(
            results_frame,
            text="Top 3 Predictions:\n--",
            style='Info.TLabel',
            justify=tk.LEFT
        )
        self.top3_label.pack(anchor='w', pady=(0, 8))
        
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
        self.spec_canvas.bind('<Configure>', lambda _e: self._draw_spec_canvas_chrome())

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


    def _draw_spec_canvas_chrome(self):
        """Draw mode-specific chrome for the spectrogram canvas."""
        if not hasattr(self, 'spec_canvas'):
            return

        canvas = self.spec_canvas
        canvas.delete('spec_chrome')
        w = max(canvas.winfo_width(), 40)
        h = max(canvas.winfo_height(), 40)
        p = self.current_palette

        if self.ui_mode_var.get() == 'Modern':
            self._create_rounded_rect(
                canvas,
                4,
                4,
                w - 4,
                h - 4,
                radius=16,
                outline=p['placeholder'],
                width=1,
                fill=p['canvas'],
                tags='spec_chrome',
            )
        else:
            canvas.create_rectangle(
                4,
                4,
                w - 4,
                h - 4,
                outline=p['placeholder'],
                width=1,
                tags='spec_chrome',
            )

        canvas.tag_lower('spec_chrome')
    
    
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
        self.top3_label.config(text="Top 3 Predictions:\n--")
        self.save_button.config(state=tk.DISABLED)
        self.batch_button.config(state=tk.DISABLED)
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
            prediction_details = self.predictor.predict_details_from_audio(
                file_path,
                mel_db=mel_db,
                mel_image=mel_image
            )

            self.root.after(
                0,
                self.update_results,
                predicted_genre,
                confidence,
                prediction_details,
                mel_image,
                mel_db.shape,
                elapsed_ms
            )
        except Exception as e:
            self.root.after(0, self.show_error, f"Processing failed: {e}")
    
    
    def update_results(self, genre, confidence, prediction_details, mel_image, mel_shape, elapsed_ms):
        """Update the results display."""
        self.processing_active = False
        self._stop_telemetry_loop()
        self.progress_bar.stop()
        self.current_genre = genre
        self.current_confidence = confidence
        self.current_mel_image = mel_image
        self.current_mel_shape = mel_shape

        is_low_confidence = bool(prediction_details.get('is_low_confidence', False))
        top3 = prediction_details.get('top3', [])

        if is_low_confidence:
            self.genre_label.config(text=f"Low confidence ({genre})", foreground=self.current_palette['warning'])
            self.status_label.config(text="Prediction is below confidence threshold", foreground=self.current_palette['warning'])
        else:
            self.genre_label.config(text=genre, foreground=self.current_palette['accent'])
            self.status_label.config(text="Mel-spectrogram generated and displayed", foreground=self.current_palette['success'])

        self.confidence_label.config(text=f"{confidence}%")
        if top3:
            lines = [f"{i + 1}. {item['genre']} ({item['confidence']}%)" for i, item in enumerate(top3[:3])]
            self.top3_label.config(text="Top 3 Predictions:\n" + "\n".join(lines))
        else:
            self.top3_label.config(text="Top 3 Predictions:\n--")
        self.metrics_label.config(text=f"Processing time: {elapsed_ms:.0f} ms | Mel shape: {mel_shape}")
        self.render_spectrogram_on_canvas(mel_image)
        self._hide_canvas_telemetry()
        self.save_button.config(state=tk.NORMAL)
        self.batch_button.config(state=tk.NORMAL)


    def _load_performance_snapshot_rows(self):
        """Load per-genre performance snapshot from CSV or fallback values."""
        snapshot_path = self.project_root / 'model_performance_snapshot.csv'
        rows = []

        if snapshot_path.exists():
            with open(snapshot_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append({
                        'genre': row.get('genre', ''),
                        'recall': row.get('recall', ''),
                        'f1': row.get('f1', ''),
                    })
            return rows

        # Fallback snapshot from latest chunk model diagnostics.
        fallback = [
            ('Electronic', '0.7830', ''),
            ('Experimental', '0.4211', ''),
            ('Folk', '0.7355', ''),
            ('Hip-Hop', '0.8114', ''),
            ('Instrumental', '0.5000', ''),
            ('International', '0.8438', ''),
            ('Pop', '0.5652', ''),
            ('Rock', '0.6250', ''),
        ]
        for genre, recall, f1 in fallback:
            rows.append({'genre': genre, 'recall': recall, 'f1': f1})
        return rows


    def show_performance_snapshot(self):
        """Open a small report-friendly window with per-genre performance."""
        rows = self._load_performance_snapshot_rows()
        win = tk.Toplevel(self.root)
        win.title('Per-Genre Performance Snapshot')
        win.geometry('520x360')

        ttk.Label(
            win,
            text='Model Performance by Genre',
            style='Subheader.TLabel'
        ).pack(pady=(12, 8))

        tree = ttk.Treeview(win, columns=('genre', 'recall', 'f1'), show='headings', height=10)
        tree.heading('genre', text='Genre')
        tree.heading('recall', text='Recall')
        tree.heading('f1', text='F1 (optional)')
        tree.column('genre', width=180)
        tree.column('recall', width=120, anchor=tk.CENTER)
        tree.column('f1', width=120, anchor=tk.CENTER)

        for row in rows:
            tree.insert('', tk.END, values=(row['genre'], row['recall'], row['f1']))

        tree.pack(fill=tk.BOTH, expand=True, padx=16, pady=8)

        note = ttk.Label(
            win,
            text='Source: model_performance_snapshot.csv (fallback values shown when file is missing).',
            style='Info.TLabel'
        )
        note.pack(pady=(0, 10))


    def run_demo_self_check(self):
        """Run a lightweight reliability pass and save a timestamped report."""
        checks = []

        checks.append(('TensorFlow available', tf is not None))
        checks.append(('Loaded model available', self.predictor.use_nn and self.predictor.model is not None))
        checks.append(('Chunk voting enabled profile', bool(self.predictor.use_chunk_voting)))
        checks.append(('Batch export button active', hasattr(self, 'batch_button')))
        checks.append(('UI mode selector available', hasattr(self, 'ui_mode_combo')))

        checkpoint_path = self.project_root / 'model_checkpoints' / 'final_chunk_retrained_resnet50.keras'
        checks.append(('Final chunk checkpoint exists', checkpoint_path.exists()))

        train_manifest = self.project_root / 'processed_data' / 'spectrograms_chunked' / 'train_manifest.csv'
        val_manifest = self.project_root / 'processed_data' / 'spectrograms_chunked' / 'val_manifest.csv'
        checks.append(('Chunk train manifest exists', train_manifest.exists()))
        checks.append(('Chunk val manifest exists', val_manifest.exists()))

        snapshot_path = self.project_root / 'model_performance_snapshot.csv'
        checks.append(('Per-genre snapshot CSV exists', snapshot_path.exists()))

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.project_root / f'demo_self_check_{timestamp}.txt'

        passed = sum(1 for _, ok in checks if ok)
        total = len(checks)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('AudioTexture Demo Self-Check Report\n')
            f.write(f'Timestamp: {datetime.now().isoformat()}\n\n')
            for name, ok in checks:
                f.write(f"[{'PASS' if ok else 'FAIL'}] {name}\n")
            f.write(f"\nSummary: {passed}/{total} checks passed\n")
            f.write('\nRecommended demo flow:\n')
            f.write('1. Single-file inference\n')
            f.write('2. Top-3 + low-confidence display\n')
            f.write('3. Batch folder export to CSV\n')
            f.write('4. UI mode switch Classic <-> Modern\n')
            f.write('5. Performance Snapshot window\n')

        color = self.current_palette['success'] if passed == total else self.current_palette['warning']
        self.status_label.config(
            text=f'Demo self-check complete: {passed}/{total} passed',
            foreground=color,
        )
        self.metrics_label.config(text=f'Report saved: {report_path.name}')


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
        self.batch_button.config(state=tk.NORMAL)


    def on_batch_infer_click(self):
        """Run folder-level batch inference and export results to CSV."""
        if self.processing_active:
            self.status_label.config(text="Processing already in progress...", foreground="#FF9800")
            return

        folder_path = filedialog.askdirectory(title="Select folder with audio files")
        if not folder_path:
            return

        csv_path = filedialog.asksaveasfilename(
            title="Save batch results CSV",
            defaultextension=".csv",
            initialfile="batch_predictions.csv",
            filetypes=[("CSV File", "*.csv")],
        )
        if not csv_path:
            return

        self.processing_active = True
        self.progress_bar.start()
        self.status_label.config(text="Running batch inference...", foreground="#FF9800")
        self.metrics_label.config(text="Batch mode: processing files")
        self.batch_button.config(state=tk.DISABLED)

        thread = threading.Thread(target=self._run_batch_inference, args=(folder_path, csv_path), daemon=True)
        thread.start()


    def _run_batch_inference(self, folder_path, csv_path):
        """Background batch inference worker."""
        valid_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg'}
        audio_files = [
            p for p in sorted(Path(folder_path).iterdir())
            if p.is_file() and p.suffix.lower() in valid_extensions
        ]

        rows = []
        for audio_path in audio_files:
            try:
                t0 = time.perf_counter()
                mel_image, mel_db = audio_to_mel_image(str(audio_path))
                details = self.predictor.predict_details_from_audio(
                    str(audio_path),
                    mel_db=mel_db,
                    mel_image=mel_image,
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                top3 = details.get('top3', [])
                rows.append({
                    'filename': audio_path.name,
                    'predicted_genre': details.get('genre', ''),
                    'confidence': details.get('confidence', ''),
                    'second_choice': top3[1]['genre'] if len(top3) > 1 else '',
                    'third_choice': top3[2]['genre'] if len(top3) > 2 else '',
                    'processing_ms': f"{elapsed_ms:.0f}",
                    'low_confidence': str(bool(details.get('is_low_confidence', False))),
                })
            except Exception as exc:
                rows.append({
                    'filename': audio_path.name,
                    'predicted_genre': '',
                    'confidence': '',
                    'second_choice': '',
                    'third_choice': '',
                    'processing_ms': '',
                    'low_confidence': '',
                    'error': str(exc),
                })

        fieldnames = [
            'filename',
            'predicted_genre',
            'confidence',
            'second_choice',
            'third_choice',
            'processing_ms',
            'low_confidence',
            'error',
        ]
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        self.root.after(0, self._on_batch_inference_complete, csv_path, len(audio_files), len(rows))


    def _on_batch_inference_complete(self, csv_path, files_count, rows_count):
        """Finalize batch-mode UI updates."""
        self.processing_active = False
        self.progress_bar.stop()
        self.batch_button.config(state=tk.NORMAL)
        self.status_label.config(
            text=f"Batch complete: {files_count} files processed",
            foreground="#4CAF50",
        )
        self.metrics_label.config(text=f"Saved CSV: {Path(csv_path).name} ({rows_count} rows)")


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
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        self._load_trained_models()


    def _build_prediction_details(self, probs):
        """Convert a probability vector into top-k prediction details."""
        probs = np.asarray(probs, dtype=np.float32)
        denom = float(np.sum(probs))
        if denom <= 0.0:
            raise RuntimeError("Invalid probabilities from model.")
        probs = probs / denom

        top_idx = np.argsort(probs)[::-1][:3]
        top3 = []
        for idx in top_idx:
            confidence = int(round(float(probs[idx]) * 100.0))
            top3.append({'genre': self.labels[int(idx)], 'confidence': max(1, min(confidence, 99))})

        top_genre = top3[0]['genre']
        top_confidence = top3[0]['confidence']
        is_low_confidence = top_confidence < self.confidence_threshold

        return {
            'genre': top_genre,
            'confidence': top_confidence,
            'top3': top3,
            'is_low_confidence': is_low_confidence,
            'threshold': self.confidence_threshold,
        }


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
        tfm = tf

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
        @tfm.keras.utils.register_keras_serializable(package='AudioTexture')
        class ApplicationPreprocess(tfm.keras.layers.Layer):
            def __init__(self, mode, **kwargs):
                super().__init__(**kwargs)
                self.mode = mode

            def call(self, inputs):
                if self.mode == 'resnet50':
                    return tfm.keras.applications.resnet.preprocess_input(inputs)
                if self.mode == 'efficientnetv2b0':
                    return tfm.keras.applications.efficientnet_v2.preprocess_input(inputs)
                raise ValueError(f'Unknown preprocess mode: {self.mode}')

            def get_config(self):
                config = super().get_config()
                config.update({'mode': self.mode})
                return config

        custom_objects = {
            'preprocess_input': tfm.keras.applications.resnet.preprocess_input,
            'ApplicationPreprocess': ApplicationPreprocess,
        }

        for name in preferred:
            ckpt_path = ckpt_dir / name
            if not ckpt_path.exists():
                continue
            try:
                self.model = tfm.keras.models.load_model(
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
        details = self._build_prediction_details(probs)
        return details['genre'], details['confidence']


    def _predict_with_models_details(self, mel_image):
        """Run neural inference and return full top-k details."""
        if self.model is None:
            raise RuntimeError("No neural model is loaded.")

        if self.input_mode == 'raw_rgb':
            x = self._prepare_raw_rgb_tensor(mel_image)
        else:
            x = self._prepare_engineered_tensor(mel_image)
        probs = self.model(x, training=False).numpy()[0].astype(np.float32)
        return self._build_prediction_details(probs)


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
        details = self._build_prediction_details(probs)
        return details['genre'], details['confidence']


    def _predict_from_audio_chunks_details(self, file_path):
        """Chunk-voting inference that returns top-k details."""
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
                fmax=8000,
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
        return self._build_prediction_details(probs)


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

        details = self._build_prediction_details(probs)
        return details['genre'], details['confidence']


    def _predict_stub_details(self, mel_db):
        """Return top-k details for deterministic fallback predictor."""
        # Feature vector derived from spectral texture
        mean_energy = float(np.mean(mel_db))
        spread = float(np.std(mel_db))
        high_band = float(np.mean(mel_db[-16:, :]))
        low_band = float(np.mean(mel_db[:16, :]))
        rhythm_proxy = float(np.std(np.diff(mel_db, axis=1)))

        features = np.array([mean_energy, spread, high_band, low_band, rhythm_proxy], dtype=np.float32)

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
        return self._build_prediction_details(probs)

    def predict_details_from_audio(self, file_path, mel_db=None, mel_image=None):
        """Return full prediction details including top-3 and low-confidence state."""
        if self.use_nn:
            if self.use_chunk_voting:
                try:
                    return self._predict_from_audio_chunks_details(file_path)
                except Exception as exc:
                    print(f"Chunk voting failed; attempting single-image fallback: {exc}")
                    if mel_db is not None:
                        try:
                            model_image = mel_db_to_model_image(mel_db)
                            return self._predict_with_models_details(model_image)
                        except Exception as exc2:
                            print(f"Single-image neural inference failed: {exc2}")
            else:
                if mel_db is not None:
                    try:
                        model_image = mel_db_to_model_image(mel_db)
                        return self._predict_with_models_details(model_image)
                    except Exception as exc:
                        print(f"Single-image neural inference failed; falling back to stub: {exc}")

        if mel_db is not None:
            return self._predict_stub_details(mel_db)

        _, mel_db_local = audio_to_mel_image(file_path)
        return self._predict_stub_details(mel_db_local)


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
