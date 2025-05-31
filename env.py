# Creating an environment class for starting or restarting the game, by scanning the play.png image on the screen.

# Importing the libraries
import numpy as np
import pyautogui
import time
from pynput import keyboard
import random
import os
from pathlib import Path

# Importing the files
from action import action
from start_game import begin
from preprocess_image import preprocess_image

class env:
    def __init__(self):
        self.action_space = 5
        # Get the base directory and images folder
        self.base_dir = Path(__file__).parent
        self.images_dir = self.base_dir / "images"
        
        # Initialize game location
        self.loc = begin()
        pyautogui.click(
            x=int(self.loc["left"] + self.loc["width"] / 2),
            y=int(self.loc["top"] + self.loc["height"] / 2),
            clicks=1,
            button='left'
        )
        self.act = action(
            int(self.loc["left"]), 
            int(self.loc["top"]), 
            int(self.loc["width"]), 
            int(self.loc["height"])
        )     
        
    # To take random action.
    def action_space_sample(self):
        return random.randint(0,4)
    
    # While play button not found, keep checking every .1 seconds.
    # If play button visible, click on it and send the image after waiting 2.5 seconds.
    def reset(self):
        play_img_path = self.images_dir / "play.png"
        
        print(f"Looking for play button at: {play_img_path}")
        
        # Check if play button image exists
        if not play_img_path.exists():
            print(f"ERROR: play.png not found at {play_img_path}")
            return None
            
        max_attempts = 100  # Prevent infinite loop
        attempts = 0
        
        while attempts < max_attempts:
            try:
                # Look for play button with different confidence levels
                play_location = None
                for confidence in [0.8, 0.7, 0.6, 0.5, 0.4]:
                    try:
                        play_location = pyautogui.locateOnScreen(
                            str(play_img_path), 
                            confidence=confidence,
                            grayscale=True
                        )
                        if play_location:
                            print(f"Found play button with confidence {confidence}")
                            break
                    except pyautogui.ImageNotFoundException:
                        continue
                
                if play_location:
                    # Click on the play button
                    x, y = pyautogui.center(play_location)
                    print(f"Clicking play button at ({x}, {y})")
                    pyautogui.click(x, y)
                    
                    # CRITICAL FIX: Wait longer for game to fully load
                    print("Waiting for game to start...")
                    time.sleep(3.0)  # Increased from 2.5 to 3.0 seconds
                    
                    # CRITICAL FIX: Clear any pending mouse/keyboard actions
                    pyautogui.PAUSE = 0.1  # Small pause between actions
                    
                    print("Game should be ready now")
                    break
                else:
                    if attempts == 0:  # Only print this once to reduce spam
                        print(f"Play button not found, attempt {attempts + 1}")
                    time.sleep(0.1)
                    attempts += 1
                    
            except Exception as e:
                print(f"Error looking for play button: {e}")
                time.sleep(0.1)
                attempts += 1
        
        if attempts >= max_attempts:
            print("ERROR: Could not find play button after maximum attempts")
            return None
        
        # CRITICAL FIX: Take screenshot AFTER game has fully loaded
        print("Capturing initial game state...")
        try:
            # Wait a bit more to ensure game is stable
            time.sleep(0.5)
            
            state = preprocess_image(pyautogui.screenshot(region=(
                int(self.loc["left"]), 
                int(self.loc["top"]), 
                int(self.loc["width"]), 
                int(self.loc["height"])
            )))
            print("Initial state captured successfully")
            return state
        except Exception as e:
            print(f"Error taking screenshot: {e}")
            return None

        # After each step check if game over.
    # Wait for .2 seconds after taking action and see if play.png is in the frame.
    # If play.png is not visible, return next_state.
    # if game over, reward = -5 else reward = 1.
    def step(self, action):
        # CRITICAL FIX: Add small delay before performing action
        time.sleep(0.1)  # Let game stabilize before acting
        
        print(f"performing action")
        self.act.perform(action)
        
        # CRITICAL FIX: Wait for action to complete before checking game state
        time.sleep(0.2)  # Wait for action to take effect
        
        Done = True
        
        play_img_path = self.images_dir / "play.png"
        
        try:
            # Check if game is still running (no play button visible)
            play_location = pyautogui.locateOnScreen(
                str(play_img_path), 
                confidence=0.4,
                grayscale=True
            )
            Done = play_location is not None
        except pyautogui.ImageNotFoundException:
            Done = False
        except Exception as e:
            print(f"Error checking game state: {e}")
            Done = True  # Assume game over if we can't check
        
        # CRITICAL FIX: Only take screenshot if game is still running
        next_state = None
        if not Done:
            try:
                next_state = preprocess_image(pyautogui.screenshot(region=(
                    int(self.loc["left"]), 
                    int(self.loc["top"]), 
                    int(self.loc["width"]), 
                    int(self.loc["height"])
                )))
            except Exception as e:
                print(f"Error taking screenshot in step: {e}")
                next_state = None
        
        reward = 2
        if Done:
            reward = -10
            print("Game Over detected")
            print("\nGame Ended\n")  # Add separator for clarity
            
        return (next_state, reward, Done, {})