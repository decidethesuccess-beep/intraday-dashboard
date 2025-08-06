# keep_alive.py
# This module runs a simple Flask web server in a separate thread
# to keep the Replit Repl alive by responding to HTTP pings.

from flask import Flask
from threading import Thread
import logging

# Configure logging for the keep_alive module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("bot_log.log"),
              logging.StreamHandler()])
logger = logging.getLogger(__name__)

app = Flask('')


@app.route('/')
def home():
    """
    Responds to a GET request to the root URL.
    This is the endpoint UptimeRobot will ping.
    """
    logger.info(
        "Keep-alive ping received. Responding 'Trading system is alive!'")
    return "Trading system is alive!", 200


def run():
    """
    Runs the Flask application.
    Host '0.0.0.0' makes it accessible externally within the Replit environment.
    Port 8080 is a common port for web servers in Replit.
    """
    logger.info("Starting keep-alive Flask server on 0.0.0.0:8080...")
    app.run(host='0.0.0.0', port=8080)


def keep_alive():
    """
    Starts the Flask server in a separate daemon thread.
    A daemon thread runs in the background and exits when the main program exits.
    """
    logger.info("Initializing keep-alive thread...")
    t = Thread(target=run)
    t.daemon = True  # Allows the main program to exit even if this thread is still running
    t.start()
    logger.info("Keep-alive thread started.")


if __name__ == '__main__':
    # This block is for testing keep_alive.py directly
    print("--- Starting keep_alive.py direct test ---")
    keep_alive()
    print("Keep-alive server should be running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(
                1)  # Keep main thread alive to allow Flask server to run
    except KeyboardInterrupt:
        print("Keep-alive test interrupted.")
    print("--- keep_alive.py direct test end ---")
