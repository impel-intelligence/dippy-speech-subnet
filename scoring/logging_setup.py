import logging
from colorama import init, Fore, Style

# Initialize Colorama for cross-platform color support
init()

def setup_logging() -> None:
    # JSON log format for production (no colors)
    log_format = (
        '{"time": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", '
        '"process": %(process)d, "message": "%(message)s"}'
    )

    logging.basicConfig(
        level=logging.DEBUG,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add custom log levels with colors
    logging.addLevelName(logging.DEBUG, f"{Fore.CYAN}DEBUG{Style.RESET_ALL}")
    logging.addLevelName(logging.INFO, f"{Fore.GREEN}INFO{Style.RESET_ALL}")
    logging.addLevelName(logging.WARNING, f"{Fore.YELLOW}WARNING{Style.RESET_ALL}")
    logging.addLevelName(logging.ERROR, f"{Fore.RED}ERROR{Style.RESET_ALL}")
    logging.addLevelName(logging.CRITICAL, f"{Fore.MAGENTA}CRITICAL{Style.RESET_ALL}")
