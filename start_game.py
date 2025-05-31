import pyautogui
import time
from pathlib import Path

def begin():
    """
    1) Looks for the initial "Tap to Play" (start_t.png) and clicks it.
    2) Then waits for the in-game "Play" button (play.png) to appear.
    3) Returns the {top, left, width, height} of the game region.
    """
    print("Beginning game setup...")
    base = Path(__file__).parent / 'images'
    start_img = base / 'start_t.png'
    play_img  = base / 'play.png'
    
    # Check if image files exist
    if not start_img.exists():
        print(f"ERROR: start_t.png not found at {start_img}")
        # Try to find any obvious game start button
        print("Attempting to find game window manually...")
        # Return a default game region (you'll need to adjust these values)
        return {'top': 100, 'left': 100, 'width': 400, 'height': 600}
    
    if not play_img.exists():
        print(f"ERROR: play.png not found at {play_img}")

    print(f"Looking for start button at: {start_img}")

    # 1) Find & click the first "Tap to Play"
    loc_start = None
    max_attempts = 50
    attempts = 0
    
    while loc_start is None and attempts < max_attempts:
        try:
            # Try different confidence levels
            for confidence in [0.8, 0.7, 0.6, 0.5]:
                try:
                    loc_start = pyautogui.locateOnScreen(
                        str(start_img),
                        confidence=confidence,
                        grayscale=True
                    )
                    if loc_start:
                        print(f"Found start button with confidence {confidence}")
                        break
                except pyautogui.ImageNotFoundException:
                    continue
                    
            if not loc_start:
                print(f"Start button not found, attempt {attempts + 1}")
                time.sleep(0.2)
                attempts += 1
                
        except Exception as e:
            print(f"Error looking for start button: {e}")
            attempts += 1
            time.sleep(0.2)
    
    if not loc_start:
        print("Could not find start button. Please manually start the game.")
        # Return a reasonable default region - adjust these coordinates for your setup
        return {'top': 100, 'left': 100, 'width': 400, 'height': 600}
    
    # Click the start button
    center = pyautogui.center(loc_start)
    print(f"Clicking start button at {center}")
    pyautogui.click(center)
    time.sleep(2.0)  # Wait longer for game to load

    # 2) Wait for the in-game "Play" button
    loc_play = None
    attempts = 0
    max_attempts = 50
    
    while loc_play is None and attempts < max_attempts:
        try:
            # Try different confidence levels
            for confidence in [0.8, 0.7, 0.6, 0.5, 0.4]:
                try:
                    loc_play = pyautogui.locateOnScreen(
                        str(play_img),
                        confidence=confidence,
                        grayscale=True
                    )
                    if loc_play:
                        print(f"Found play button with confidence {confidence}")
                        break
                except pyautogui.ImageNotFoundException:
                    continue
                    
            if not loc_play:
                print(f"Play button not found, attempt {attempts + 1}")
                time.sleep(0.2)
                attempts += 1
                
        except Exception as e:
            print(f"Error looking for play button: {e}")
            attempts += 1
            time.sleep(0.2)

    if not loc_play:
        print("Could not find play button. Using start button location as reference.")
        # Use start button location to estimate game area
        top = loc_start.top + loc_start.height
        left = loc_start.left
        width = loc_start.width
        height = 400  # Estimate height
        return {'top': top, 'left': left, 'width': width, 'height': height}

    # 3) Compute the game window rectangle more accurately
    # Use both start and play button positions to determine game area
    top = min(loc_start.top, loc_play.top)
    left = min(loc_start.left, loc_play.left)
    
    right = max(loc_start.left + loc_start.width, loc_play.left + loc_play.width)
    bottom = max(loc_start.top + loc_start.height, loc_play.top + loc_play.height)
    
    width = right - left
    height = bottom - top
    
    print(f"Game area detected: top={top}, left={left}, width={width}, height={height}")
    
    return {'top': top, 'left': left, 'width': width, 'height': height}