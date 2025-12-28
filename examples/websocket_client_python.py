"""Python WebSocket client example for MCP Hub.

This example shows how to connect to the MCP Hub WebSocket server
and receive real-time progress updates for long-running operations.
"""

import asyncio
import json
import websockets


async def stream_operation(operation_id: str):
    """Stream updates for a specific operation.

    Args:
        operation_id: ID of the operation to monitor
    """
    uri = "ws://localhost:8765"

    async with websockets.connect(uri) as websocket:
        # Receive welcome message
        welcome = await websocket.recv()
        print(f"Connected: {welcome}\n")

        # Subscribe to operation
        subscribe_cmd = {
            "command": "subscribe",
            "operation_id": operation_id
        }
        await websocket.send(json.dumps(subscribe_cmd))

        # Receive subscription confirmation
        response = await websocket.recv()
        print(f"Subscription: {response}\n")

        # Listen for events
        print(f"Listening for updates on operation {operation_id}...\n")

        try:
            async for message in websocket:
                event = json.loads(message)

                event_type = event.get("event_type")
                data = event.get("data", {})

                if event_type == "progress":
                    progress = data.get("progress", 0)
                    message_text = data.get("message", "")
                    print(f"[PROGRESS {progress*100:.0f}%] {message_text}")

                elif event_type == "status":
                    status = data.get("status")
                    message_text = data.get("message", "")
                    print(f"[STATUS] {status}: {message_text}")

                elif event_type == "log":
                    level = data.get("level")
                    log_message = data.get("message")
                    print(f"[LOG:{level.upper()}] {log_message}")

                elif event_type == "result":
                    result = data.get("result")
                    print(f"\n[RESULT]\n{json.dumps(result, indent=2)}")

                elif event_type == "completed":
                    duration = data.get("duration")
                    print(f"\n[COMPLETED] Operation finished in {duration:.2f}s")
                    break

                elif event_type == "error":
                    error = data.get("error")
                    error_type = data.get("error_type")
                    print(f"\n[ERROR:{error_type}] {error}")
                    break

        except websockets.exceptions.ConnectionClosed:
            print("\nConnection closed")


async def monitor_all_operations():
    """Monitor all operations (no subscription filter).

    This receives all events from all operations.
    """
    uri = "ws://localhost:8765"

    async with websockets.connect(uri) as websocket:
        # Receive welcome message
        welcome = await websocket.recv()
        print(f"Connected: {welcome}\n")
        print("Monitoring all operations...\n")

        try:
            async for message in websocket:
                event = json.loads(message)

                operation_id = event.get("operation_id")
                event_type = event.get("event_type")
                data = event.get("data", {})

                if event_type == "progress":
                    progress = data.get("progress", 0)
                    message_text = data.get("message", "")
                    print(f"[{operation_id[:8]}] PROGRESS {progress*100:.0f}%: {message_text}")

                elif event_type == "status":
                    status = data.get("status")
                    print(f"[{operation_id[:8]}] STATUS: {status}")

                elif event_type == "result":
                    print(f"[{operation_id[:8]}] RESULT RECEIVED")

                elif event_type == "completed":
                    duration = data.get("duration")
                    print(f"[{operation_id[:8]}] COMPLETED in {duration:.2f}s")

                elif event_type == "error":
                    error_type = data.get("error_type")
                    print(f"[{operation_id[:8]}] ERROR: {error_type}")

        except websockets.exceptions.ConnectionClosed:
            print("\nConnection closed")


async def get_server_stats():
    """Get WebSocket server statistics."""
    uri = "ws://localhost:8765"

    async with websockets.connect(uri) as websocket:
        # Receive welcome message
        await websocket.recv()

        # Request stats
        stats_cmd = {"command": "get_stats"}
        await websocket.send(json.dumps(stats_cmd))

        # Receive stats
        response = await websocket.recv()
        stats = json.loads(response)

        print("WebSocket Server Statistics:")
        print(json.dumps(stats, indent=2))


async def interactive_monitor():
    """Interactive operation monitor with user input."""
    uri = "ws://localhost:8765"

    async with websockets.connect(uri) as websocket:
        welcome = await websocket.recv()
        print(f"Connected: {json.loads(welcome)['data']['message']}\n")

        print("Commands:")
        print("  subscribe <operation_id> - Subscribe to operation updates")
        print("  stats - Show server statistics")
        print("  quit - Exit\n")

        while True:
            command = input("> ").strip()

            if command == "quit":
                break

            elif command == "stats":
                await websocket.send(json.dumps({"command": "get_stats"}))
                response = await websocket.recv()
                print(json.dumps(json.loads(response), indent=2))

            elif command.startswith("subscribe "):
                operation_id = command.split(" ", 1)[1]
                await websocket.send(json.dumps({
                    "command": "subscribe",
                    "operation_id": operation_id
                }))
                print(f"Subscribed to {operation_id}")

                # Listen for updates
                try:
                    async for message in websocket:
                        event = json.loads(message)
                        print(f"Event: {event['event_type']} - {event['data']}")

                        if event['event_type'] in ['completed', 'error']:
                            break
                except (json.JSONDecodeError, KeyError):
                    # Ignore malformed messages
                    pass

            else:
                print("Unknown command")


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "monitor":
            # Monitor all operations
            asyncio.run(monitor_all_operations())
        elif sys.argv[1] == "stats":
            # Get server stats
            asyncio.run(get_server_stats())
        elif sys.argv[1] == "interactive":
            # Interactive mode
            asyncio.run(interactive_monitor())
        else:
            # Stream specific operation
            operation_id = sys.argv[1]
            asyncio.run(stream_operation(operation_id))
    else:
        print("Usage:")
        print("  python websocket_client_python.py <operation_id>  - Stream specific operation")
        print("  python websocket_client_python.py monitor         - Monitor all operations")
        print("  python websocket_client_python.py stats           - Get server stats")
        print("  python websocket_client_python.py interactive     - Interactive mode")
