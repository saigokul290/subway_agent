# fast_capture.py
# ──────────────────────────────────────────────────────────────────────────────
# Continuously grabs the defined screen rectangle at ~30–60 FPS using mss.
# Your main env.step() will call grabber.get_frame() to fetch the latest frame.
# ──────────────────────────────────────────────────────────────────────────────

import mss
import numpy as np
import threading

class FastFrameGrabber:
    def __init__(self, bbox):
        """
        bbox: {'top': y, 'left': x, 'width': w, 'height': h}
        describing the region of the screen to capture (game window).
        """
        self.bbox = bbox
        self.sct = mss.mss()
        self.latest_frame = None
        self._stop_flag = False
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()

    def _grab_loop(self):
        while not self._stop_flag:
            sct_img = self.sct.grab({
                "top": self.bbox["top"],
                "left": self.bbox["left"],
                "width": self.bbox["width"],
                "height": self.bbox["height"],
            })
            arr = np.array(sct_img)  # BGRA
            self.latest_frame = arr[:, :, :3]  # keep BGR
            # no sleep: capture as fast as possible

    def get_frame(self):
        """
        Return the most recent BGR image as a numpy array (or None if not yet ready).
        """
        return self.latest_frame

    def stop(self):
        self._stop_flag = True
        self._thread.join()
        self.sct.close()
