"""Start the dashboard:  python run.py  ->  http://127.0.0.1:5000"""
import argparse

from queue_predictor.app import app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Queue Collapse Predictor dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to open it to other devices")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
