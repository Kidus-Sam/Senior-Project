"""
Checkpoint 1 - Deliverable 2: AudioTexture GUI Shell with Drag-and-Drop
This is the initial TkinterDnD2 shell that receives dragged files and displays a placeholder Results layout.
"""

import tkinter as tk
from tkinter import ttk
import threading
from pathlib import Path

# Try to import TkinterDnD2
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    HAS_DND = True
except ImportError:
    print("Warning: tkinterdnd2 not installed. Drag-and-drop will be limited.")
    HAS_DND = False
    TkinterDnD = None


class AudioTextureGUI:
    """Main GUI class for AudioTexture application."""
    
    def __init__(self, root):
        """Initialize the AudioTexture GUI."""
        self.root = root
        self.root.title("AudioTexture - Music Genre Classification")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)
        
        # Current state
        self.current_file = None
        self.current_genre = None
        self.current_confidence = None
        
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
        if HAS_DND:
            self.drop_canvas.drop_target_register(DND_FILES)
            self.drop_canvas.dnd_bind('<<Drop>>', self.on_file_drop)
            self.drop_canvas.dnd_bind('<<DragEnter>>', self.on_drag_enter)
            self.drop_canvas.dnd_bind('<<DragLeave>>', self.on_drag_leave)
    
    
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
        
        # Simulate processing (in real implementation, this would trigger the pipeline)
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
        
        # Spectrogram canvas placeholder
        ttk.Label(results_frame, text="Mel-Spectrogram Visualization:", 
                 style='Info.TLabel').pack(pady=(20, 10))
        
        self.spec_canvas = tk.Canvas(results_frame, bg="#f5f5f5", 
                                     highlightbackground="#CCCCCC",
                                     height=150)
        self.spec_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Draw placeholder
        self.spec_canvas.create_text(self.spec_canvas.winfo_width() // 2, 75,
                                    text="[Spectrogram will display here]",
                                    fill="#CCCCCC", font=('Segoe UI', 12))
    
    
    def start_processing(self, file_path):
        """Start processing the file in a background thread."""
        self.progress_bar.start()
        self.status_label.config(text="Processing audio...", foreground="#FF9800")
        
        # Run processing in background thread
        thread = threading.Thread(target=self.process_file, args=(file_path,))
        thread.daemon = True
        thread.start()
    
    
    def process_file(self, file_path):
        """Simulate file processing (placeholder)."""
        import time
        
        # Simulate processing time
        time.sleep(2)
        
        # Parse filename to generate placeholder results
        file_name = Path(file_path).stem
        
        # Placeholder genre (in real implementation, this comes from the model)
        genres = ['Electronic', 'Hip-Hop', 'Rock', 'Pop', 'Folk', 'Experimental', 
                 'Instrumental', 'International']
        predicted_genre = genres[hash(file_name) % len(genres)]
        confidence = 75 + (hash(file_name) % 20)
        
        # Update UI from main thread
        self.root.after(0, self.update_results, predicted_genre, confidence)
    
    
    def update_results(self, genre, confidence):
        """Update the results display."""
        self.progress_bar.stop()
        self.current_genre = genre
        self.current_confidence = confidence
        
        self.genre_label.config(text=genre)
        self.confidence_label.config(text=f"{confidence}%")
        self.status_label.config(text="Classification complete!", foreground="#4CAF50")
    
    
    def show_error(self, message):
        """Display an error message."""
        self.status_label.config(text=f"Error: {message}", foreground="#F44336")


def main():
    """Main entry point."""
    if HAS_DND:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    app = AudioTextureGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
