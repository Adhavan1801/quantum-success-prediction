# Entrypoint to run the Quantum Success Prediction Web Application
import os
import sys

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.server import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f"  QUANTUM SUCCESS PREDICTION PLATFORM")
    print(f"  Web UI & API running locally at: http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(host="127.0.0.1", port=port, debug=False)
