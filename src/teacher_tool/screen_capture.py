import mss
import numpy as np
import cv2
import time

class ScreenCapture:
    def __init__(self, monitor_index=1):
        """
        Initialize Screen Capture parameters.
        MSS instance is created per-capture to ensure thread safety.
        """
        self.monitor_index = monitor_index
        self.roi = None
        
        # Get monitor details once to set defaults (using temporary mss)
        with mss.mss() as sct:
            if len(sct.monitors) > monitor_index:
                self.monitor = sct.monitors[monitor_index]
            else:
                print(f"Monitor {monitor_index} not found. Using primary.")
                self.monitor = sct.monitors[1]

    def select_roi(self):
        """
        Allows user to select a region of interest using OpenCV interactive selection.
        """
        print("Capturing snapshot for ROI selection...")
        with mss.mss() as sct:
            screenshot = sct.grab(self.monitor)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        print("Select the region for Zoom/Meet window and press ENTER or SPACE.")
        r = cv2.selectROI("Select Region", img, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Region")

        if r[2] > 0 and r[3] > 0:
            self.roi = {
                'top': int(self.monitor['top'] + r[1]),
                'left': int(self.monitor['left'] + r[0]),
                'width': int(r[2]),
                'height': int(r[3])
            }
            print(f"ROI selected: {self.roi}")
        else:
            print("No ROI selected. Capturing full screen.")
            self.roi = None

    def capture(self):
        """
        Captures a frame using a local MSS instance (Thread-safe).
        """
        region = self.roi if self.roi else self.monitor
        
        # Create new MSS instance for every capture to avoid threading issues
        # MSS is lightweight, so this is acceptable for stability
        with mss.mss() as sct:
            screenshot = sct.grab(region)
            img = np.array(screenshot)
            
        # MSS returns BGRA, convert to BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def release(self):
        pass # Nothing to release as we use context managers

if __name__ == "__main__":
    cap = ScreenCapture()
    print("Press 'r' to select region, 'q' to quit.")
    while True:
        frame = cap.capture()
        cv2.imshow("Screen Capture Test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('r'): cap.select_roi()
    cv2.destroyAllWindows()