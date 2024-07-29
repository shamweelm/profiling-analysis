# Setup a logger to be used in the module
import logging
import os

class Logger:
    def __init__(self):
        log_file = os.getenv('LOG_FILE', 'logs/app.log')
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()

        # Convert log level from string to logging constant
        numeric_level = getattr(logging, log_level, logging.INFO)

        # Create a logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(numeric_level)

        # Create a file handler
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)

        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)

        # Create a formatter and set it for both handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add the handlers to the logger
        self.logger.addHandler(file_handler)
        # self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger

# # Example usage:
# if __name__ == "__main__":
#     log = Logger().get_logger()
#     log.debug('This is a debug message')
#     log.info('This is an info message')
#     log.warning('This is a warning message')
#     log.error('This is an error message')
#     log.critical('This is a critical message')
