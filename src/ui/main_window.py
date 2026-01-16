import sys
import os
import customtkinter as ctk
from tkinter import messagebox, filedialog
import threading
import time
import matplotlib
matplotlib.use('TkAgg') 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import cv2
import numpy as np
from PIL import Image, ImageTk
import csv
from datetime import datetime

# --- IMPORT SERVICES ---
# (Assumes app is run from project root via 'python main.py')
try:
    from src.data.repository import EngagementDB
    # Note: 'main_app' logic is now merged/replaced or handled differently. 
    # But wait, lines 343 "from main_app import TeacherMonitorApp" refers to the OLD Logic file.
    # We must REWIRE this to use 'face_service' and 'capture_service' DIRECTLY 
    # or create a new Controller class here to replace TeacherMonitorApp.
except ImportError:
    # Fallback for direct execution (testing only)
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.data.repository import EngagementDB

# --- OCEAN BREEZE PALETTE (Refined) ---
COLOR_BG = "#F1F5F9"       # Slate 100 (Main Background)
COLOR_SIDEBAR = "#FFFFFF"  # White (Sidebar)
COLOR_CARD = "#FFFFFF"     # White (Cards)
COLOR_ACCENT = "#0EA5E9"   # Sky 500
COLOR_ACCENT_HOVER = "#0284C7" # Sky 600
COLOR_TEXT = "#0F172A"     # Slate 900
COLOR_TEXT_MUTED = "#64748B" # Slate 500
COLOR_BORDER = "#E2E8F0"   # Slate 200
COLOR_SUCCESS = "#10B981"  # Emerald 500
COLOR_WARNING = "#F59E0B"  # Amber 500
COLOR_DANGER = "#EF4444"   # Red 500

ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

class StatCard(ctk.CTkFrame):
    """Custom Widget for KPI Cards"""
    def __init__(self, master, title, icon_char, value_color=COLOR_TEXT):
        super().__init__(master, fg_color=COLOR_CARD, corner_radius=16, border_width=1, border_color=COLOR_BORDER)
        self.grid_columnconfigure(0, weight=1)
        
        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=16, pady=(16, 0))
        
        ctk.CTkLabel(header, text=icon_char, font=("Segoe UI Emoji", 18)).pack(side="left")
        ctk.CTkLabel(header, text=title, font=("Inter", 12, "bold"), text_color=COLOR_TEXT_MUTED).pack(side="left", padx=8)
        
        # Value
        self.value_lbl = ctk.CTkLabel(self, text="--", font=("Inter", 28, "bold"), text_color=value_color)
        self.value_lbl.pack(anchor="w", padx=16, pady=(8, 4))
        
        # Subtitle/Trend
        self.sub_lbl = ctk.CTkLabel(self, text="Waiting...", font=("Inter", 11), text_color=COLOR_TEXT_MUTED)
        self.sub_lbl.pack(anchor="w", padx=16, pady=(0, 16))

    def update_value(self, value, sub_text=None):
        self.value_lbl.configure(text=str(value))
        if sub_text:
            self.sub_lbl.configure(text=sub_text)

class ModernTeacherApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Teacher Station Pro - v2.0")
        self.geometry("1400x900")
        self.configure(fg_color=COLOR_BG)
        
        # --- App State ---
        self.is_monitoring = False
        self.monitor_thread = None
        self.app_core = None
        self.current_stats = None
        self.alert_enabled = False
        self.session_start_time = None
        
        # --- Layout Grid ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 1. SIDEBAR
        self.setup_sidebar()

        # 2. MAIN CONTAINER
        self.container = ctk.CTkFrame(self, fg_color=COLOR_BG, corner_radius=0)
        self.container.grid(row=0, column=1, sticky="nsew")
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        # Frames map
        self.frames = {}
        self.active_frame = None

        # Initialize Frames
        self.frames["dashboard"] = self.create_dashboard_frame()
        self.frames["analytics"] = self.create_analytics_frame()
        self.frames["settings"] = self.create_settings_frame()

        self.show_frame("dashboard")
        
        # Lazy Load AI
        self.after(500, self.lazy_load_core)

    def setup_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=260, corner_radius=0, fg_color=COLOR_SIDEBAR, border_width=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        # Sidebar Right Border (Fake shadow)
        border = ctk.CTkFrame(self.sidebar, width=1, fg_color=COLOR_BORDER)
        border.place(relx=1.0, rely=0, relheight=1.0, anchor="ne")

        # Logo Area
        logo_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        logo_frame.pack(fill="x", padx=24, pady=32)
        ctk.CTkLabel(logo_frame, text="⚡", font=("Segoe UI Emoji", 32)).pack(side="left")
        
        title_box = ctk.CTkFrame(logo_frame, fg_color="transparent")
        title_box.pack(side="left", padx=12)
        ctk.CTkLabel(title_box, text="Insight", font=("Inter", 20, "bold"), text_color=COLOR_TEXT).pack(anchor="w")
        ctk.CTkLabel(title_box, text="Classroom AI", font=("Inter", 12), text_color=COLOR_TEXT_MUTED).pack(anchor="w")

        # Navigation
        self.nav_btns = {}
        items = [
            ("dashboard", "Dashboard", "🏠"),
            ("analytics", "Analytics", "📊"),
            ("settings", "Settings", "⚙️")
        ]
        
        for key, text, icon in items:
            btn = ctk.CTkButton(
                self.sidebar, text=f"  {icon}   {text}", 
                height=50, anchor="w",
                fg_color="transparent", text_color=COLOR_TEXT_MUTED,
                hover_color=COLOR_BG, font=("Inter", 14, "bold"),
                corner_radius=8,
                command=lambda k=key: self.show_frame(k)
            )
            btn.pack(fill="x", padx=16, pady=4)
            self.nav_btns[key] = btn

        # Bottom Profile/Status
        status_card = ctk.CTkFrame(self.sidebar, fg_color=COLOR_BG, corner_radius=12)
        status_card.pack(side="bottom", fill="x", padx=16, pady=24)
        
        self.lbl_status_dot = ctk.CTkLabel(status_card, text="●", font=("Arial", 16), text_color=COLOR_WARNING)
        self.lbl_status_dot.pack(side="left", padx=(16, 8), pady=12)
        self.lbl_status_text = ctk.CTkLabel(status_card, text="System Init...", font=("Inter", 12, "bold"), text_color=COLOR_TEXT)
        self.lbl_status_text.pack(side="left", pady=12)

    def create_dashboard_frame(self):
        frame = ctk.CTkFrame(self.container, fg_color="transparent")
        
        # Grid Layout: 3 Columns. Col 0+1 = Video (66%), Col 2 = Stats (33%)
        frame.grid_columnconfigure(0, weight=1) 
        frame.grid_columnconfigure(1, weight=1) 
        frame.grid_columnconfigure(2, weight=1) # Side panel
        frame.grid_rowconfigure(1, weight=1) # Main content

        # --- TOP HEADER ---
        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=3, sticky="ew", padx=32, pady=(32, 24))
        
        ctk.CTkLabel(header, text="Live Monitoring", font=("Inter", 26, "bold"), text_color=COLOR_TEXT).pack(side="left")
        self.lbl_timer = ctk.CTkLabel(header, text="00:00:00", font=("Monospace", 16), text_color=COLOR_TEXT_MUTED)
        self.lbl_timer.pack(side="right")

        # --- LEFT: VIDEO PREVIEW (Spans 2 columns) ---
        video_container = ctk.CTkFrame(frame, fg_color=COLOR_CARD, corner_radius=20, border_width=1, border_color=COLOR_BORDER)
        video_container.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=(32, 16), pady=(0, 32))
        
        # Video "Live" Badge
        self.live_badge = ctk.CTkLabel(video_container, text=" OFF AIR ", fg_color=COLOR_BORDER, text_color=COLOR_TEXT_MUTED, corner_radius=6, font=("Inter", 10, "bold"), width=60)
        self.live_badge.place(x=20, y=20)

        self.video_label = ctk.CTkLabel(video_container, text="", text_color=COLOR_TEXT_MUTED)
        self.video_label.pack(fill="both", expand=True, padx=2, pady=40)
        
        self.hero_text = ctk.CTkLabel(video_container, text="Click 'Start Session'\nto activate camera feed.", font=("Inter", 16), text_color=COLOR_TEXT_MUTED)
        self.hero_text.place(relx=0.5, rely=0.5, anchor="center")

        # --- RIGHT: STATS & CONTROLS ---
        side_panel = ctk.CTkFrame(frame, fg_color="transparent")
        side_panel.grid(row=1, column=2, sticky="nsew", padx=(0, 32), pady=(0, 32))

        # 1. Action Buttons
        self.btn_start = ctk.CTkButton(side_panel, text="▶  Start Session", height=50, fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER, font=("Inter", 15, "bold"), corner_radius=12, command=self.start_monitoring)
        self.btn_start.pack(fill="x", pady=(0, 12))
        
        self.btn_stop = ctk.CTkButton(side_panel, text="⏹  Stop Session", height=50, fg_color=COLOR_CARD, border_width=1, border_color=COLOR_BORDER, hover_color="#FEE2E2", text_color=COLOR_DANGER, font=("Inter", 15, "bold"), corner_radius=12, command=self.stop_monitoring, state="disabled")
        self.btn_stop.pack(fill="x", pady=(0, 24))

        # 2. KPI Grid (2x2)
        kpi_grid = ctk.CTkFrame(side_panel, fg_color="transparent")
        kpi_grid.pack(fill="x")
        kpi_grid.grid_columnconfigure(0, weight=1)
        kpi_grid.grid_columnconfigure(1, weight=1)

        self.card_score = StatCard(kpi_grid, "Focus Score", "🎯", COLOR_ACCENT)
        self.card_score.grid(row=0, column=0, padx=(0, 8), pady=(0, 16), sticky="ew")

        self.card_students = StatCard(kpi_grid, "Students", "👥", COLOR_TEXT)
        self.card_students.grid(row=0, column=1, padx=(8, 0), pady=(0, 16), sticky="ew")

        self.card_emotion = StatCard(kpi_grid, "Emotion", "😊", COLOR_SUCCESS)
        self.card_emotion.grid(row=1, column=0, padx=(0, 8), pady=(0, 16), sticky="ew")
        
        self.card_gender = StatCard(kpi_grid, "M / F Ratio", "⚤", COLOR_TEXT)
        self.card_gender.grid(row=1, column=1, padx=(8, 0), pady=(0, 16), sticky="ew")

        # 3. Recent Activity / Alerts
        activity_box = ctk.CTkFrame(side_panel, fg_color=COLOR_CARD, corner_radius=16, border_width=1, border_color=COLOR_BORDER)
        activity_box.pack(fill="both", expand=True)
        
        ctk.CTkLabel(activity_box, text="Live Insights", font=("Inter", 14, "bold"), text_color=COLOR_TEXT).pack(anchor="w", padx=20, pady=16)
        
        self.activity_log = ctk.CTkTextbox(activity_box, font=("Consolas", 12), text_color=COLOR_TEXT_MUTED, fg_color="transparent")
        self.activity_log.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self.log_activity("System ready.")

        return frame

    def log_activity(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.activity_log.configure(state="normal")
        self.activity_log.insert("0.0", f"[{timestamp}] {msg}\n")
        self.activity_log.configure(state="disabled")

    # --- Analytics & Settings (Simplified for brevity but styled) ---
    def create_analytics_frame(self):
        frame = ctk.CTkFrame(self.container, fg_color="transparent")
        # Reuse existing logic but with better styling...
        # For now, just a placeholder structure similar to before
        ctk.CTkLabel(frame, text="Session Analytics", font=("Inter", 28, "bold")).pack(padx=40, pady=40, anchor="w")
        
        content = ctk.CTkFrame(frame, fg_color=COLOR_CARD, corner_radius=16, border_width=1, border_color=COLOR_BORDER)
        content.pack(fill="both", expand=True, padx=40, pady=(0, 40))

        # Matplotlib Area
        self.fig = Figure(figsize=(8, 6), dpi=100, facecolor=COLOR_CARD)
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(223)
        self.ax3 = self.fig.add_subplot(224)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=content)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=20)
        
        # Refresh Button floating
        btn = ctk.CTkButton(frame, text="Refresh Data", fg_color=COLOR_SIDEBAR, text_color=COLOR_TEXT, command=self.load_history_data)
        btn.place(relx=0.95, rely=0.05, anchor="ne")
        
        return frame

    def create_settings_frame(self):
        frame = ctk.CTkFrame(self.container, fg_color="transparent")
        ctk.CTkLabel(frame, text="Settings", font=("Inter", 28, "bold")).pack(padx=40, pady=40, anchor="w")
        
        sett_card = ctk.CTkFrame(frame, fg_color=COLOR_CARD, corner_radius=16, border_width=1, border_color=COLOR_BORDER)
        sett_card.pack(fill="x", padx=40)
        
        self.sw_alert = ctk.CTkSwitch(sett_card, text="Audible Alert (Beep) when Focus < 30%", font=("Inter", 14), progress_color=COLOR_ACCENT, command=self.toggle_alert)
        self.sw_alert.pack(padx=24, pady=24, anchor="w")
        
        ctk.CTkButton(sett_card, text="Export CSV Report", fg_color=COLOR_ACCENT, command=self.export_csv).pack(padx=24, pady=(0, 24), anchor="w")
        
        return frame

    # --- LOGIC ---
    def lazy_load_core(self):
        try:
            # CHANGED: Import from new service layer
            from src.services.app_core import AppCore
            self.app_core = AppCore()
            self.lbl_status_text.configure(text="AI Ready", text_color=COLOR_SUCCESS)
            self.lbl_status_dot.configure(text_color=COLOR_SUCCESS)
            self.btn_start.configure(state="normal")
            self.load_history_data()
        except Exception as e:
            self.lbl_status_text.configure(text="AI Error", text_color=COLOR_DANGER)
            self.lbl_status_dot.configure(text_color=COLOR_DANGER)
            self.log_activity(f"Error loading AI: {e}")
            print(f"Detailed Error: {e}") # Print to console for debug

    def show_frame(self, name):
        for k, btn in self.nav_btns.items():
            if k == name:
                btn.configure(fg_color=COLOR_BG, text_color=COLOR_ACCENT)
            else:
                btn.configure(fg_color="transparent", text_color=COLOR_TEXT_MUTED)
        
        if self.active_frame: self.active_frame.grid_forget()
        self.active_frame = self.frames[name]
        self.active_frame.grid(row=0, column=0, sticky="nsew")

    def start_monitoring(self):
        self.is_monitoring = True
        self.session_start_time = time.time()
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal", fg_color=COLOR_CARD)
        self.live_badge.configure(fg_color=COLOR_DANGER, text=" ● LIVE ", text_color="white")
        self.hero_text.place_forget()
        self.log_activity("Session started.")
        
        self.app_core.start_monitoring()
        self.monitor_thread = threading.Thread(target=self.loop, daemon=True)
        self.monitor_thread.start()
        
        self.update_timer()

    def stop_monitoring(self):
        self.is_monitoring = False
        self.app_core.stop_monitoring()
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled", fg_color=COLOR_CARD)
        self.live_badge.configure(fg_color=COLOR_BORDER, text=" OFF AIR ", text_color=COLOR_TEXT_MUTED)
        self.video_label.configure(image=None, text="Session Paused")
        self.hero_text.place(relx=0.5, rely=0.5, anchor="center")
        self.log_activity("Session ended. Data saved.")
        messagebox.showinfo("Saved", "Session data saved to Database.")
        self.load_history_data()

    def update_timer(self):
        if self.is_monitoring and self.session_start_time:
            elapsed = int(time.time() - self.session_start_time)
            h = elapsed // 3600
            m = (elapsed % 3600) // 60
            s = elapsed % 60
            self.lbl_timer.configure(text=f"{h:02}:{m:02}:{s:02}")
            self.after(1000, self.update_timer)

    def update_video_feed(self, frame_rgb, w, h):
        if not self.is_monitoring: return
        
        # Resize logic inside main thread or pre-calculated
        # To avoid lag, we accept pre-resized or resize here
        target_h = 600
        scale = target_h/h
        new_w, new_h = int(w*scale), target_h
        
        # Create image in main thread context
        try:
            pil_img = Image.fromarray(frame_rgb)
            ctk_img = ctk.CTkImage(pil_img, size=(new_w, new_h))
            
            # Update label
            self.video_label.configure(image=ctk_img, text="")
            # Keep reference to avoid garbage collection
            self.video_label.image = ctk_img 
        except Exception as e:
            print(f"Error updating video feed: {e}")

    def loop(self):
        while self.is_monitoring:
            try:
                frame, stats = self.app_core.process_frame()
                if frame is not None:
                    # Prepare data for main thread
                    h, w = frame.shape[:2]
                    # Convert color in background thread to save main thread time
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Schedule update on main thread
                    # We pass the numpy array (frame_rgb) which is thread-safe copy
                    self.after(0, self.update_video_feed, frame_rgb, w, h)
                
                if stats:
                    self.current_stats = stats
                    self.after(0, self.update_ui_stats)
                    
                    # Smart Alert Beep
                    if self.alert_enabled and stats['avg_engagement'] < 0.3:
                        print('\a') 
            except Exception as e:
                print(f"Error in monitoring loop: {e}")

            time.sleep(0.03) # Cap at ~30 FPS to reduce CPU usage

    def update_ui_stats(self):
        if not self.current_stats: return
        s = self.current_stats
        
        self.card_score.update_value(f"{s['avg_engagement']:.2f}", "Avg Focus Level")
        self.card_students.update_value(str(s['total_students']), "Detected Faces")
        
        # Emotion
        emotions = {k:v for k,v in s.items() if k in ['Happy','Sad','Neutral','Angry','Surprise']}
        if emotions:
            dom = max(emotions, key=emotions.get)
            self.card_emotion.update_value(dom, "Dominant Mood")
            
            # Simple Log Logic (Example)
            if dom == "Sad" or dom == "Angry":
                # Avoid spamming log
                pass
                
        # Gender
        self.card_gender.update_value(f"{s.get('male_count',0)}/{s.get('female_count',0)}", "Male / Female")

    def load_history_data(self):
        if not self.app_core: return
        print("[GUI] Loading Analytics Data...")
        
        try:
            # 1. Fetch all sessions (Newest first)
            sessions = self.app_core.db.get_all_sessions()
            if not sessions: 
                print("[GUI] No sessions found.")
                return
            
            # 2. Get latest session ID
            latest_session = sessions[0]
            last_id = latest_session[0]
            print(f"[GUI] Visualizing Session ID: {last_id}")
            
            # 3. Fetch logs for this session
            logs = self.app_core.db.get_session_logs(last_id)
            
            # 4. Clear old plots
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            for ax in [self.ax1, self.ax2, self.ax3]:
                ax.set_facecolor(COLOR_CARD)

            if not logs:
                self.ax1.text(0.5, 0.5, "No Data (Session too short?)", ha='center', va='center')
                print("[GUI] Session has 0 logs.")
            else:
                # 5. Process Data
                eng = [row[4] for row in logs]
                # Smooth data if enough points
                if len(eng) > 5: 
                    eng = np.convolve(eng, np.ones(3)/3, mode='valid') # Smaller kernel
                
                self.ax1.plot(eng, color=COLOR_ACCENT, linewidth=2)
                self.ax1.set_title(f"Focus Trend (Session #{last_id})", fontsize=10)
                self.ax1.grid(True, linestyle='--', alpha=0.3)
                
                # Gender Pie
                m = np.mean([row[7] for row in logs])
                f = np.mean([row[8] for row in logs])
                if m+f > 0: 
                    self.ax2.pie([m, f], labels=['M', 'F'], colors=['#3b82f6', '#ec4899'], autopct='%1.0f%%')
                    self.ax2.set_title("Avg Gender Ratio", fontsize=9)
                
                # Emotion Bar
                emo_counts = {}
                for row in logs:
                    if row[9]:
                        parts = row[9].split(',')
                        for p in parts:
                            if ':' in p:
                                k,v = p.split(':')
                                emo_counts[k] = emo_counts.get(k,0) + int(v)
                if emo_counts:
                    self.ax3.bar(emo_counts.keys(), emo_counts.values(), color=COLOR_ACCENT)
                    self.ax3.set_title("Emotion Distribution", fontsize=9)
                    self.ax3.tick_params(axis='x', labelsize=8, rotation=15)

            # 6. Force Redraw
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            print(f"[GUI ERROR] Failed to load history: {e}")

    def toggle_alert(self):
        self.alert_enabled = bool(self.sw_alert.get())

    def export_csv(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv")
        if path:
            sessions = self.app_core.db.get_all_sessions()
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "Start", "End", "Students", "AvgEngagement", "Status"])
                writer.writerows(sessions)
            messagebox.showinfo("Export", "Done!")

if __name__ == "__main__":
    app = ModernTeacherApp()
    app.mainloop()