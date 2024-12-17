import os
import logging
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
    Function to log and print "Hello World".
    """
    logging.info("Executing job: log_and_print_hello")  # Execution confirmation
    message = "Hello World"
    logging.info(message)  # Logs the message
    print(message, flush=True)  # Print and flush immediately to avoid buffering

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
