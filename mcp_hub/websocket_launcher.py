"""WebSocket server launcher for running alongside Gradio.

This module provides utilities to launch the WebSocket server in a background
thread or as a separate process.
"""

import asyncio
import threading
from typing import Optional
from .logging_config import logger

try:
    from .websocket_server import start_websocket_server, WEBSOCKETS_AVAILABLE
except ImportError:
    WEBSOCKETS_AVAILABLE = False


class WebSocketServerThread(threading.Thread):
    """Thread for running WebSocket server alongside Gradio."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        """Initialize WebSocket server thread.

        Args:
            host: Host to bind to
            port: Port to bind to
        """
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def run(self):
        """Run the WebSocket server."""
        if not WEBSOCKETS_AVAILABLE:
            logger.error("WebSocket server cannot start: websockets library not installed")
            return

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
            self.loop.run_until_complete(start_websocket_server(self.host, self.port))
        except Exception as e:
            logger.error(f"WebSocket server error: {e}")
        finally:
            self.loop.close()

    def stop(self):
        """Stop the WebSocket server."""
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)


# Global WebSocket server thread
_websocket_thread: Optional[WebSocketServerThread] = None


def start_websocket_background(host: str = "0.0.0.0", port: int = 8765):
    """Start WebSocket server in background thread.

    Args:
        host: Host to bind to
        port: Port to bind to

    Returns:
        WebSocketServerThread instance
    """
    global _websocket_thread

    if not WEBSOCKETS_AVAILABLE:
        logger.warning("WebSocket server not started: websockets library not installed")
        logger.warning("Install with: pip install websockets>=12.0")
        return None

    if _websocket_thread is not None and _websocket_thread.is_alive():
        logger.warning("WebSocket server already running")
        return _websocket_thread

    _websocket_thread = WebSocketServerThread(host, port)
    _websocket_thread.start()

    logger.info(f"WebSocket server started in background: ws://{host}:{port}")
    return _websocket_thread


def stop_websocket_background():
    """Stop background WebSocket server."""
    global _websocket_thread

    if _websocket_thread is not None:
        _websocket_thread.stop()
        _websocket_thread = None
        logger.info("WebSocket server stopped")


def is_websocket_running() -> bool:
    """Check if WebSocket server is running.

    Returns:
        True if running, False otherwise
    """
    return _websocket_thread is not None and _websocket_thread.is_alive()


def get_websocket_status() -> dict:
    """Get WebSocket server status.

    Returns:
        Dictionary with status information
    """
    from .websocket_server import ws_manager

    return {
        "available": WEBSOCKETS_AVAILABLE,
        "running": is_websocket_running(),
        "stats": ws_manager.get_stats() if is_websocket_running() else None
    }
