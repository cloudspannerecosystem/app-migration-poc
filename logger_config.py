# Copyright 2024 Google Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging


def setup_logger(name=__name__):
    # Define a custom log format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_datefmt = "%Y-%m-%d %H:%M:%S"

    # Create a logger instance
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture all log messages

    # Create a file handler for logging to a file
    file_handler = logging.FileHandler("gemini_migration.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format, log_datefmt))

    # Create a console handler for logging to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format, log_datefmt))
    console_handler.setLevel(logging.ERROR)  # Adjust level as needed

    # Add the handlers to the logger
    if not logger.hasHandlers():  # Prevent adding multiple handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
