# action.py
# ──────────────────────────────────────────────────────────────────────────────
# Now all movement in‐game is done by arrow‐key presses only.
# We have removed any click or drag inside perform().
#
#   t == 0 → do nothing       (sleep 0.30 s)
#   t == 1 → JUMP  (press “up”)
#   t == 2 → ROLL  (press “down”)
#   t == 3 → LEFT  (press “left”)
#   t == 4 → RIGHT (press “right”)
#
# No mouse movement or clicks occur inside perform().
# ──────────────────────────────────────────────────────────────────────────────

import numpy as np
import pyautogui
import time

class action:
    """
    Encapsulates a single in‐game move.  Only arrow keys (no clicks/swipes).
    """

    def __init__(self, left, top, width, height):
        # We still store these original values in case env.reset() wants them,
        # but perform() itself will no longer click at all.
        self.left   = left
        self.top    = top
        self.width  = width
        self.height = height

    def perform(self, t):
        """
        Execute exactly one of {0..4}.  No mouse movement—purely keyboard.
        If t is a NumPy array, convert to int.
        """

        if isinstance(t, np.ndarray):
            t = int(t.item())

        # 0 → do nothing for 0.30s
        if t == 0:
            time.sleep(0.30)
            return

        # 1 → JUMP   (Up arrow)
        # 2 → ROLL   (Down arrow)
        # 3 → LEFT   (Left arrow)
        # 4 → RIGHT  (Right arrow)
        if t == 1:
            pyautogui.press('up')
        elif t == 2:
            pyautogui.press('down')
        elif t == 3:
            pyautogui.press('left')
        elif t == 4:
            pyautogui.press('right')
        else:
            # If somehow out‐of‐range, just wait
            time.sleep(0.30)
            return

        # Allow a brief pause so the game processes the keypress
        time.sleep(0.20)
