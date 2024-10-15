import warnings


class ColoredLog:
    # ANSI escape sequences for color definitions
    COLORS = {
        "INFO": "\033[92m",  # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "RESET": "\033[0m",  # Reset to default color
    }

    @staticmethod
    def info(message):
        """Displays an informational message in green."""
        ColoredLog._print(message, "INFO")

    @staticmethod
    def warning(message):
        """Displays a warning message in yellow."""
        ColoredLog._print(message, "WARNING")

    @staticmethod
    def error(message):
        """Displays an error message in red."""
        ColoredLog._print(message, "ERROR")

    @staticmethod
    def _print(message, log_type):
        """Outputs a colored message.

        Args:
            message (str): The message to display.
            log_type (str): Type of log ('INFO', 'WARNING', 'ERROR').
        """
        # Get the color based on the log type, default to yellow if not found
        color = ColoredLog.COLORS.get(log_type, ColoredLog.COLORS["WARNING"])
        # Format the message with the chosen color and reset color at the end
        formatted_message = f"{color}{message}{ColoredLog.COLORS['RESET']}"
        # Use the built-in warnings.warn to display the formatted message
        warnings.warn(formatted_message, stacklevel=2)


# Override the default formatwarning method to ensure consistent output format
def custom_format_warning(msg, *args, **kwargs):
    return f"{msg}\n"


warnings.formatwarning = custom_format_warning

if __name__ == "__main__":

    # Usage examples
    ColoredLog.info("This is an INFO.")
    ColoredLog.warning("This is WARNING.")
    ColoredLog.error("This is ERROR.")
