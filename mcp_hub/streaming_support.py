"""Streaming support for agents with WebSocket integration.

This module provides mixins and utilities for agents to stream progress
updates in real-time through WebSocket connections.
"""

from typing import Optional, Any, Dict
from contextlib import asynccontextmanager

try:
    from .websocket_server import ws_manager
    WEBSOCKET_SUPPORT = True
except ImportError:
    WEBSOCKET_SUPPORT = False
    ws_manager = None


class StreamingMixin:
    """Mixin class to add streaming capabilities to agents.

    Add this mixin to agent classes to enable real-time progress streaming.

    Example:
        class MyAgent(StreamingMixin):
            async def process(self, data):
                async with self.stream_operation('process', {'input': data}):
                    await self.emit_progress(0.2, "Processing started")
                    # ... do work ...
                    await self.emit_progress(0.8, "Almost done")
                    result = await self.finish()
                    return result
    """

    def __init__(self, *args, **kwargs):
        """Initialize streaming mixin."""
        super().__init__(*args, **kwargs)
        self._current_operation_id: Optional[str] = None

    @asynccontextmanager
    async def stream_operation(
        self,
        operation_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Context manager for streaming an operation.

        Args:
            operation_type: Type of operation
            metadata: Optional metadata

        Yields:
            Operation ID
        """
        if not WEBSOCKET_SUPPORT or ws_manager is None:
            # Fallback: no streaming
            yield None
            return

        try:
            # Create operation
            self._current_operation_id = await ws_manager.create_operation(
                operation_type,
                metadata
            )

            yield self._current_operation_id

        except Exception as e:
            # Emit error
            if self._current_operation_id:
                await ws_manager.emit_error(
                    self._current_operation_id,
                    str(e),
                    type(e).__name__
                )
            raise

        finally:
            self._current_operation_id = None

    async def emit_progress(
        self,
        progress: float,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Emit progress update for current operation.

        Args:
            progress: Progress value (0.0 to 1.0)
            message: Progress message
            details: Optional additional details
        """
        if not WEBSOCKET_SUPPORT or ws_manager is None:
            return

        if self._current_operation_id:
            await ws_manager.emit_progress(
                self._current_operation_id,
                progress,
                message,
                details
            )

    async def emit_status(
        self,
        status: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Emit status update for current operation.

        Args:
            status: Status value
            message: Status message
            details: Optional additional details
        """
        if not WEBSOCKET_SUPPORT or ws_manager is None:
            return

        if self._current_operation_id:
            await ws_manager.emit_status(
                self._current_operation_id,
                status,
                message,
                details
            )

    async def emit_log(self, level: str, message: str):
        """Emit log message for current operation.

        Args:
            level: Log level (info, warning, error)
            message: Log message
        """
        if not WEBSOCKET_SUPPORT or ws_manager is None:
            return

        if self._current_operation_id:
            await ws_manager.emit_log(
                self._current_operation_id,
                level,
                message
            )

    async def emit_result(self, result: Any):
        """Emit final result for current operation.

        Args:
            result: Operation result
        """
        if not WEBSOCKET_SUPPORT or ws_manager is None:
            return

        if self._current_operation_id:
            await ws_manager.emit_result(
                self._current_operation_id,
                result
            )


async def stream_orchestration(
    user_request: str,
    orchestrator_func,
    *args,
    **kwargs
) -> Any:
    """Stream an orchestration operation with progress updates.

    Args:
        user_request: User's request
        orchestrator_func: Orchestrator function to call
        *args: Additional positional arguments
        **kwargs: Additional keyword arguments

    Returns:
        Orchestration result
    """
    if not WEBSOCKET_SUPPORT or ws_manager is None:
        # No streaming support, just call function
        return await orchestrator_func(user_request, *args, **kwargs)

    # Create operation
    operation_id = await ws_manager.create_operation(
        "orchestration",
        {"user_request": user_request}
    )

    try:
        # Step 1: Question enhancement
        await ws_manager.emit_progress(
            operation_id,
            0.1,
            "Enhancing question...",
            {"step": "question_enhancement"}
        )

        # Step 2: Web search
        await ws_manager.emit_progress(
            operation_id,
            0.3,
            "Searching the web...",
            {"step": "web_search"}
        )

        # Step 3: LLM processing
        await ws_manager.emit_progress(
            operation_id,
            0.5,
            "Processing with LLM...",
            {"step": "llm_processing"}
        )

        # Step 4: Code generation
        await ws_manager.emit_progress(
            operation_id,
            0.7,
            "Generating code...",
            {"step": "code_generation"}
        )

        # Step 5: Code execution
        await ws_manager.emit_progress(
            operation_id,
            0.9,
            "Executing code...",
            {"step": "code_execution"}
        )

        # Execute orchestration
        result = await orchestrator_func(user_request, *args, **kwargs)

        # Emit result
        await ws_manager.emit_result(operation_id, result)

        return result

    except Exception as e:
        await ws_manager.emit_error(
            operation_id,
            str(e),
            type(e).__name__
        )
        raise


def create_progress_callback(operation_id: str):
    """Create a progress callback function for an operation.

    Args:
        operation_id: Operation ID

    Returns:
        Async callback function
    """
    async def callback(progress: float, message: str, details: Optional[Dict] = None):
        """Progress callback."""
        if WEBSOCKET_SUPPORT and ws_manager:
            await ws_manager.emit_progress(operation_id, progress, message, details)

    return callback


def create_log_callback(operation_id: str):
    """Create a log callback function for an operation.

    Args:
        operation_id: Operation ID

    Returns:
        Async callback function
    """
    async def callback(level: str, message: str):
        """Log callback."""
        if WEBSOCKET_SUPPORT and ws_manager:
            await ws_manager.emit_log(operation_id, level, message)

    return callback


class ProgressTracker:
    """Helper class for tracking multi-step progress.

    Example:
        tracker = ProgressTracker(operation_id, total_steps=5)
        await tracker.step("Step 1", details={"info": "xyz"})
        await tracker.step("Step 2")
        # ... etc
    """

    def __init__(self, operation_id: str, total_steps: int):
        """Initialize progress tracker.

        Args:
            operation_id: Operation ID
            total_steps: Total number of steps
        """
        self.operation_id = operation_id
        self.total_steps = total_steps
        self.current_step = 0

    async def step(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Advance to next step and emit progress.

        Args:
            message: Step message
            details: Optional additional details
        """
        if not WEBSOCKET_SUPPORT or ws_manager is None:
            return

        self.current_step += 1
        progress = self.current_step / self.total_steps

        await ws_manager.emit_progress(
            self.operation_id,
            progress,
            message,
            details
        )

    async def complete(self, result: Any):
        """Mark as complete and emit result.

        Args:
            result: Final result
        """
        if not WEBSOCKET_SUPPORT or ws_manager is None:
            return

        await ws_manager.emit_result(self.operation_id, result)


async def run_with_progress(
    operation_type: str,
    func,
    steps: list,
    metadata: Optional[Dict[str, Any]] = None
):
    """Run a function with automatic progress tracking.

    Args:
        operation_type: Type of operation
        func: Async function to run
        steps: List of step descriptions
        metadata: Optional metadata

    Returns:
        Function result
    """
    if not WEBSOCKET_SUPPORT or ws_manager is None:
        return await func()

    operation_id = await ws_manager.create_operation(operation_type, metadata)

    try:
        tracker = ProgressTracker(operation_id, len(steps))

        # Emit each step (in practice, you'd call these at appropriate times)
        for step_msg in steps:
            await tracker.step(step_msg)

        result = await func()

        await tracker.complete(result)

        return result

    except Exception as e:
        await ws_manager.emit_error(operation_id, str(e), type(e).__name__)
        raise
