import os
import logging
import subprocess
from apscheduler.schedulers.blocking import BlockingScheduler

# Ensure log directory exists
log_dir = "/app/logs"
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Logs to console (stdout)
    ],
    force=True  # Ensures no prior configs interfere
)

def log_and_print_hello():
    """
    Function to log and print "Hello World" via a subprocess.
    """
    logging.info("Executing job: log_and_print_hello in subprocess")  # Execution confirmation

    # Command to print Hello World
    command = ["python", "-c", "print('Hello World')"]

    try:
        # Run the command in a subprocess and capture output
        result = subprocess.run(command, text=True, capture_output=True, check=True)
        
        # Log subprocess output
        logging.info(f"Subprocess Output: {result.stdout.strip()}")
        print(result.stdout.strip(), flush=True)  # Print and flush subprocess output
    except subprocess.CalledProcessError as e:
        logging.error(f"Subprocess failed: {e.stderr}")
        print(f"Error: {e.stderr}", flush=True)

def main():
    # Create an instance of scheduler
    scheduler = BlockingScheduler()

    # Add job to scheduler: run `log_and_print_hello` every 10 seconds (for testing)
    scheduler.add_job(log_and_print_hello, 'interval', seconds=10, id='hello_world_job')

    try:
        logging.info("Scheduler started.")
        print("Scheduler started.", flush=True)  # Flush print
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logging.info("Scheduler stopped.")
        print("Scheduler stopped.", flush=True)

if __name__ == "__main__":
    main()
