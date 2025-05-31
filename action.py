# Hybrid control: Mouse for game setup, Keyboard for gameplay

import pyautogui
import time
import numpy as np

class GameSetup():
    """Handles game initialization using mouse controls"""
    
    def __init__(self):
        pyautogui.PAUSE = 0.01
        
    def find_and_click_button(self, image_path, confidence=0.7):
        """Find and click a button using image recognition"""
        try:
            button_location = pyautogui.locateOnScreen(image_path, confidence=confidence)
            if button_location:
                center = pyautogui.center(button_location)
                print(f"Found button at: {center}")
                pyautogui.click(center)
                return True
            else:
                print(f"Button not found: {image_path}")
                return False
        except Exception as e:
            print(f"Error finding button: {e}")
            return False
    
    def setup_game(self, start_button_path, play_button_path=None):
        """Setup the game by clicking start and play buttons"""
        print("Setting up game...")
        
        # Click start button
        if self.find_and_click_button(start_button_path):
            time.sleep(1)  # Wait for game to load
            
            # Click play button if provided
            if play_button_path:
                self.find_and_click_button(play_button_path)
                time.sleep(0.5)
            
            print("Game setup complete!")
            return True
        return False

class GameController():
    """Handles gameplay using keyboard controls only"""
    
    def __init__(self):
        # No coordinates needed for keyboard control
        pyautogui.PAUSE = 0.01
        pyautogui.FAILSAFE = False
        
    def perform(self, action_code):
        """Perform game action using keyboard only"""
        # Convert numpy array or any other type to int
        if isinstance(action_code, np.ndarray):
            action_code = int(action_code.item())

        print(f"Performing keyboard action: {action_code}")
        
        if action_code == 0:  # Do nothing
            time.sleep(0.1)
            return
            
        # Define keyboard actions using arrow keys
        keyboard_actions = {
            1: 'up',        # Jump (up arrow)
            2: 'down',      # Roll/Duck (down arrow)
            3: 'left',      # Move Left (left arrow)
            4: 'right',     # Move Right (right arrow)
        }
        
        if action_code in keyboard_actions:
            key = keyboard_actions[action_code]
            pyautogui.press(key)
            time.sleep(0.05)  # Short delay to allow game to respond
        else:
            print(f"Unknown action: {action_code}")

# Legacy compatibility class for existing code
class action():
    """Compatibility wrapper that ignores coordinates and uses keyboard"""
    
    def __init__(self, left=0, top=0, width=0, height=0):
        # Ignore all coordinate parameters
        print("Action class initialized - using keyboard controls (coordinates ignored)")
        self.controller = GameController()
        
    def perform(self, action_code):
        """Delegate to keyboard controller"""
        self.controller.perform(action_code)

# Example usage showing the separation:
if __name__ == "__main__":
    # Step 1: Game Setup (Mouse control)
    setup = GameSetup()
    game_ready = setup.setup_game(
        start_button_path="images/start_button.png",
        play_button_path="images/play_button.png"
    )
    
    if game_ready:
        # Step 2: Gameplay (Keyboard control)
        controller = GameController()
        
        print("Starting gameplay with keyboard controls...")
        # Test gameplay actions
        controller.perform(1)  # Jump
        time.sleep(0.5)
        controller.perform(3)  # Left
        time.sleep(0.5)
        controller.perform(4)  # Right
        time.sleep(0.5)
        controller.perform(2)  # Duck
        
        print("Gameplay test complete!")
    else:
        print("Game setup failed!")