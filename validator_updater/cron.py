import os
import logging
import subprocess
import shlex
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

def log_and_run_validator():
    """
    Function to log and run 'validator.py' via a subprocess.
    """
    logging.info("Executing job: Running validator.py in subprocess")  # Execution confirmation

    # Retrieve the environment variable
    validator_command = os.environ.get("VALIDATOR_COMMAND")

    if not validator_command:
        raise ValueError("VALIDATOR_COMMAND environment variable is not set.")
    
    # Split the command safely into a list
    command = shlex.split(validator_command)

    try:
        # Run the validator.py script and stream its output in real-time
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Stream stdout in real-time
        for stdout_line in iter(process.stdout.readline, ''):
            logging.info(stdout_line.strip())

        # Wait for the process to complete and capture any remaining stderr
        _, stderr = process.communicate()

        if stderr:
            logging.error(f"Subprocess Error:\n{stderr.strip()}")

        if process.returncode != 0:
            logging.error(f"Subprocess exited with return code {process.returncode}")
        else:
            logging.info("Subprocess completed successfully.")
    except Exception as e:
        logging.error(f"Error running subprocess: {str(e)}")

def main():
    # Create an instance of scheduler
    scheduler = BlockingScheduler()

    # Add job to scheduler: run `log_and_run_validator` at the beginning of every hour and during the mid point of every hour  
    scheduler.add_job(log_and_run_validator, 'cron',  minute='0, 30', id='validator_job')


    try:
        logging.info("Scheduler started.")
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logging.info("Scheduler stopped.")

if __name__ == "__main__":
    main()
