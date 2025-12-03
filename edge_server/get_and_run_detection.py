"""
Cross-platform script to download model and run detection.
Can be executed directly: python get_and_run_detection.py
Or piped from curl: curl -X POST http://localhost:8000/trigger_train | python
"""
import os
import sys
import subprocess
import urllib.request

SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:8000")

def download_file(url, filename):
    """Download a file from the server and return response headers."""
    try:
        print(f"Downloading {filename}...")
        request = urllib.request.Request(url)
        response = urllib.request.urlopen(request)
        
        # Get model name from header if available
        model_name = response.headers.get('X-Model-Name', 'unknown')
        
        with open(filename, 'wb') as f:
            f.write(response.read())
        
        if os.path.exists(filename):
            print(f"Successfully downloaded {filename}")
            if model_name != 'unknown':
                print(f"Model name: {model_name}")
            return True, model_name
        else:
            print(f"Error: {filename} was not downloaded")
            return False, None
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return False, None

def main():
    print(f"Connecting to server: {SERVER_URL}")
    
    # Download the model
    model_url = f"{SERVER_URL}/download_latest_model"
    success, model_name = download_file(model_url, "best.pt")
    if not success:
        print(f"Failed to download model from {SERVER_URL}")
        print("Make sure:")
        print("  1. The server is running on the target machine")
        print("  2. The SERVER_URL is correct (use SERVER_URL=http://IP:8000)")
        print("  3. Firewall allows connections on port 8000")
        sys.exit(1)
    
    # Download the detection script
    script_url = f"{SERVER_URL}/download_detection_script"
    success, _ = download_file(script_url, "run_detection.py")
    if not success:
        print(f"Failed to download detection script from {SERVER_URL}")
        sys.exit(1)
    
    # Set model name as environment variable for run_detection.py
    if model_name and model_name != 'unknown':
        os.environ['MODEL_NAME'] = model_name
    
    # Run the detection script
    print("Starting detection...")
    try:
        subprocess.run([sys.executable, "run_detection.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running detection: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDetection stopped by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()

