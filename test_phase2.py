import requests
import io
import time

def test_phase2():
    print("Testing Phase 2 Features...")
    
    # 1. API Health
    try:
        r = requests.get("http://localhost:8000/health")
        print(f"Health: {r.status_code}")
    except Exception as e:
        print(f"Health Check Failed: {e}")
        return

    # 2. Test Predict (Regression Test)
    try:
        print("Testing Prediction...")
        files = {'file': open('data/train/airplane/0.png', 'rb')}
        r = requests.post("http://localhost:8000/predict", files=files)
        print(f"Predict Status: {r.status_code}")
        print(f"Predict Response: {r.json()}")
    except Exception as e:
        print(f"Predict Failed: {e}")

    # 3. Test Explain (New Feature)
    try:
        print("Testing Explanation (Grad-CAM)...")
        files = {'file': open('data/train/airplane/0.png', 'rb')}
        r = requests.post("http://localhost:8000/explain", files=files)
        print(f"Explain Status: {r.status_code}")
        if r.status_code == 200:
            with open("gradcam_result.png", "wb") as f:
                f.write(r.content)
            print("Saved gradcam_result.png")
        else:
            print(f"Explain Error: {r.text}")
    except Exception as e:
        print(f"Explain Failed: {e}")

    # 4. Check MLflow availability
    try:
        print("Checking MLflow...")
        r = requests.get("http://localhost:5000")
        print(f"MLflow UI Status: {r.status_code}")
    except Exception as e:
        print(f"MLflow Check Failed: {e}")

if __name__ == "__main__":
    time.sleep(5) # Give services a moment
    test_phase2()
