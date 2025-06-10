# env.py
# ──────────────────────────────────────────────────────────────────────────────
# Click “play.png” to start each episode, then always use MSS for frames.
# Once the game is running, step() will never perform any mouse click— 
# it only calls action.perform(...), which is now 100% keyboard.
# Updated so that every non‐crash action yields the same base reward (+2.0).
# ──────────────────────────────────────────────────────────────────────────────

import numpy as np
import mss
import pyautogui
from pyautogui import ImageNotFoundException
import time
import cv2
from pathlib import Path

from action import action
from start_game import begin
from preprocess_image import preprocess_image  # your existing preprocessing logic

class Env:
    def __init__(self):
        self.action_space = 5

        # Base directory and images folder
        self.base_dir    = Path(__file__).parent
        self.images_dir  = self.base_dir / "images"

        # 1) “Tap to Play” / “Play” sequence via begin()
        self.loc = begin()
        left = int(self.loc["left"])
        top  = int(self.loc["top"])
        w    = int(self.loc["width"])
        h    = int(self.loc["height"])

        # Compute original bottom (top + h)
        original_bottom = top + h

        # 2) Expand upward by exactly h, clamp at 0
        new_top = top - h
        if new_top < 0:
            new_top = 0

        # 3) New height so we don’t extend below original bottom
        new_height = original_bottom - new_top

        print(f"[env.__init__] begin() returned:   left={left}, top={top},    width={w}, height={h}")
        print(f"[env.__init__] Expanded bbox →    left={left}, top={new_top}, width={w}, height={new_height}")

        self.bbox = {
            "top":    new_top,
            "left":   left,
            "width":  w,
            "height": new_height
        }

        # 4) Prepare MSS
        self.sct = mss.mss()

        # 5) Instantiate the action performer (pure keyboard now)
        self.act = action(left, top, w, h)

        # 6) Episode counter (no longer used for debug images, but kept for compatibility)
        self.episode_counter = 0

    def action_space_sample(self):
        return np.random.randint(0, self.action_space)

    def reset(self):
        """
        1) Look for 'play.png' up to 10× (0.10s apart). If found, click it up to 3× until it's gone.
           If never found, skip straight to capture.
        2) Grab one MSS frame of self.bbox, preprocess it, and return that state.
        """
        play_img = self.images_dir / "play.png"
        match_center = None

        # (A) Try up to 10× to find play.png
        for _ in range(10):
            try:
                found = pyautogui.locateOnScreen(str(play_img), confidence=0.70, grayscale=True)
            except ImageNotFoundException:
                found = None

            if found:
                match_center = pyautogui.center(found)
                break
            time.sleep(0.10)

        # (B) If found, click up to 3× until gone
        if match_center:
            for click_count in range(1, 4):
                px, py = match_center
                pyautogui.moveTo(px, py)
                pyautogui.click()
                time.sleep(0.20)

                try:
                    still_there = pyautogui.locateOnScreen(
                        str(play_img), confidence=0.70, grayscale=True
                    )
                except ImageNotFoundException:
                    still_there = None

                if still_there is None:
                    break

                match_center = pyautogui.center(still_there)

            # (C) If still there after 3 clicks, wait until gone
            while True:
                try:
                    still_there = pyautogui.locateOnScreen(
                        str(play_img), confidence=0.70, grayscale=True
                    )
                except ImageNotFoundException:
                    still_there = None

                if still_there is None:
                    break
                time.sleep(0.05)

        # (D) Grab one MSS frame of the expanded bbox
        try:
            sct_img   = self.sct.grab(self.bbox)
            frame_bgr = np.array(sct_img)[:, :, :3]
        except Exception as e:
            print(f"[env.reset] ERROR grabbing frame: {e}")
            return None

        # (E) Convert BGR→RGB, preprocess, and return
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            state = preprocess_image(rgb)
        except Exception as e:
            print(f"[env.reset] ERROR in preprocess_image: {e}")
            return None

        return state

    def step(self, action_idx, step_count):
        """
        1) Perform the action (arrow key or no-op), wait 0.02s.
        2) Grab MSS frame of self.bbox.
        3) Check full‐screen for 'play.png' only if step_count>=1 → done_flag.
        4) Compute reward:
             • step_count == 0 → always +0.50 (skip crash check on frame 0)
             • step_count >= 1:
                 – if crash (play.png seen) → reward = base - 5
                 – else → reward = base
             Here, base = +2.0 for every non‐crash action.
        5) Preprocess next_state and return (next_state, reward, done_flag, {}).
        """
        # 1) Perform action (pure keyboard)
        self.act.perform(action_idx)
        time.sleep(0.02)

        # 2) Grab MSS frame
        try:
            sct_img   = self.sct.grab(self.bbox)
            frame_bgr = np.array(sct_img)[:, :, :3]
        except Exception:
            return (None, -5.0, True, {})

        # 3) Determine done_flag
        if step_count == 0:
            # Always assume alive on frame 0
            done_flag = False
        else:
            # step_count >= 1: look for “play.png” on full screen
            done_flag = False
            play_img = self.images_dir / "play.png"
            try:
                found = pyautogui.locateOnScreen(str(play_img), confidence=0.70, grayscale=True)
            except ImageNotFoundException:
                found = None

            if found:
                done_flag = True

        # 4) Compute reward
        if step_count == 0:
            reward = +0.50
        else:
            # base reward is +2.0 for any safe (non‐crash) action
            base = 2.0
            if done_flag:
                reward = base - 5.0   # crash penalty
            else:
                reward = base

        # 5) Preprocess next_state
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            next_state = preprocess_image(rgb)
        except Exception:
            next_state = None

        return (next_state, reward, done_flag, {})

    def close(self):
        """
        Clean up MSS when the program ends.
        """
        try:
            self.sct.close()
        except Exception:
            pass
