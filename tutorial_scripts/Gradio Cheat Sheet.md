# Gradio Quick Reference Cheat Sheet

This cheat sheet provides a quick overview of Gradio concepts, components, and best practices, especially useful for building UIs for Python functions and MCP servers.

---

## 1. Core Concepts

- **`gr.Interface`**: The primary way to create a UI for a single Python function.
- **`gr.Blocks`**: A more flexible way to build complex UIs with custom layouts, multiple functions, and interactive components. Essential for your multi-agent MCP hub.
- **Function (`fn`)**: The Python function you want to expose. Gradio handles input/output.
- **Input Components (`inputs`)**: UI elements for users to provide data (e.g., `gr.Textbox`, `gr.Image`).
- **Output Components (`outputs`)**: UI elements to display results (e.g., `gr.Textbox`, `gr.Label`).
- **`launch()`**: Method to start the Gradio web server and make your UI accessible.

---

## 2. Basic App Structure

### Using `gr.Interface` (for a single function)

```python
import gradio as gr

def my_python_function(input_text):
    return f"Processed: {input_text.upper()}"

iface = gr.Interface(
    fn=my_python_function,
    inputs=gr.Textbox(label="Input Text"),
    outputs=gr.Textbox(label="Output Text"),
    title="Simple Text Processor",
    description="Enter some text to see it processed."
)

if __name__ == "__main__":
    iface.launch()
```

### Using `gr.Blocks` (for more complex UIs / multiple functions)

```python
import gradio as gr

def greet(name):
    return f"Hello, {name}!"

def farewell(name):
    return f"Goodbye, {name}!"

with gr.Blocks(title="My Multi-Function App", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Main Application Title")
    with gr.Row():
        name_input = gr.Textbox(label="Enter Name")
    with gr.Row():
        greet_button = gr.Button("Greet")
        farewell_button = gr.Button("Say Farewell")
    output_display = gr.Textbox(label="Result")

    greet_button.click(fn=greet, inputs=name_input, outputs=output_display)
    farewell_button.click(fn=farewell, inputs=name_input, outputs=output_display)

if __name__ == "__main__":
    demo.launch()
```

---

## 3. Key Gradio Components & Parameters

### `gr.Interface` Parameters

- **`fn`**: The Python function to call.
- **`inputs`**: A single Gradio component or a list of components.
- **`outputs`**: A single Gradio component or a list of components.
- **`title` (str)**: Title for the UI.
- **`description` (str)**: Markdown/HTML description.
- **`article` (str)**: Markdown/HTML content displayed below the interface.
- **`examples` (list of lists)**: Sample inputs to show users.

  ```python
  examples=[["example text 1"], ["example text 2"]]
  ```

- **`live` (bool)**: If True, updates automatically on input change (no submit button). Default: False.
- **`api_name` (str)**: Crucial for MCP. The name this function will have in the API (e.g., `/gradio_api/run/<api_name>`). If `api_name` is not provided (defaulting to None) or explicitly set to None, the interface will not be exposed as an MCP tool.
- **`allow_flagging` (str)**: Options like "auto", "manual", "never".
- **`theme` (`gr.themes.*`)**: Apply a pre-built theme (e.g., `gr.themes.Soft()`, `gr.themes.Monochrome()`).

### `gr.Blocks` Layout Elements

Used within a `with gr.Blocks() as demo:` block:

- **`gr.Row()`**: Arranges components horizontally.
- **`gr.Column()`**: Arranges components vertically (default layout behavior).
- **`gr.Tab("Tab Name")`**: Creates a tab.
- **`gr.Group()`**: Visually groups components.
- **`gr.Accordion("Title")`**: A collapsible section.
- **`gr.Markdown("## Text")`**: Display markdown.
- **`gr.HTML("<p>HTML</p>")`**: Display HTML.

### Common Input/Output Components

Most components share parameters like `label` (str) and `interactive` (bool, for inputs).

#### Text

```python
gr.Textbox(lines=1, placeholder="Enter text...", label="My Text", value="Default")
# gr.TextArea(...): Alias for gr.Textbox with lines > 1 often implied.
```

#### Numbers

```python
gr.Number(label="Quantity", value=10)
gr.Slider(minimum=0, maximum=100, step=1, label="Percentage", value=50)
```

#### Choices

```python
gr.Dropdown(choices=["Option A", "Option B"], label="Select One", value="Option A")
gr.Radio(choices=["Yes", "No"], label="Confirm", value="Yes")
gr.Checkbox(label="Agree to terms", value=False)
gr.CheckboxGroup(choices=["Red", "Green", "Blue"], label="Colors")
```

#### Files/Media

```python
gr.Image(type="pil", label="Upload Image")
# type: "pil" (Pillow Image), "numpy" (NumPy array), "filepath" (string path).
# Can also display images in output.

gr.Video(label="Upload Video")
gr.Audio(type="filepath", label="Upload Audio")
# type: "filepath", "numpy".

gr.File(label="Upload File")  # for generic files
```

#### Data Display (Primarily Outputs)

```python
gr.Label(label="Classification")  # good for showing class probabilities
gr.JSON(label="API Response")
gr.Dataframe(headers=["Col1", "Col2"], label="My Table")
gr.HighlightedText(label="Text Analysis")  # for NER, Q&A highlighting
gr.Markdown(label="Formatted Output")
gr.HTML(label="Custom HTML Output")
```

#### Buttons & Events (within `gr.Blocks`):

```python
gr.Button("Click Me", variant="primary")  # variant: "primary", "secondary", "stop"
```

**Event Listeners:** Connect component events to functions.

```python
button.click(fn=my_func, inputs=[input1, input2], outputs=[output1])
textbox.change(fn=my_func, ...)  # triggers on value change
textbox.submit(fn=my_func, ...)  # triggers on Enter key in textbox
```

---

## 4. Launching the App

```python
# For gr.Interface
iface.launch(share=False, server_name="0.0.0.0", mcp_server=False)

# For gr.Blocks
demo.launch(share=False, server_name="0.0.0.0", mcp_server=False)
```

- `share=True`: Creates a temporary public Gradio link (useful for sharing). Expires after 72 hours.
- `server_name="0.0.0.0"`: Makes the app accessible on your local network (e.g., from another device on the same Wi-Fi). Default is 127.0.0.1 (localhost).
- `server_port` (int): Specify a port (e.g., 7861).
- `mcp_server=True`: Crucial for your project. Exposes functions (from `gr.Interface` with `api_name` set) as MCP tools.
- `auth=("username", "password")`: Adds basic authentication.
- `debug=True`: Enables more detailed error messages.

---

## 5. Important Considerations for MCP

### Function Signatures

**Type Hints are VITAL:** Gradio uses them to generate the MCP schema for input/output types.

```python
def process_data(text: str, count: int) -> dict:
    # ...
    return {"result": "...", "count": count}
```

**Docstrings are VITAL:** Gradio uses them for the tool's description in the MCP schema. Use a clear format (e.g., Google style).

```python
def my_mcp_tool(param1: str) -> str:
    """
    This is the description of my MCP tool. It explains what the tool does.

    Args:
        param1 (str): Description of the first parameter.

    Returns:
        str: Description of the output.
    """
    return f"Processed {param1}"
```

### `api_name` in `gr.Interface`

- This string becomes the tool's identifier in the MCP system.
- Example: `gr.Interface(..., api_name="my_text_processor_tool")`
- If `api_name` is not provided (defaulting to None) or explicitly set to None, the interface will not be exposed as an MCP tool.

### MCP Schema Access

- Once `mcp_server=True` is active, the schema is usually at:  
  `http://<your_server_address>:<port>/gradio_api/mcp/schema`

### MCP Tool Invocation

- Tools are typically invoked via POST requests to:  
  `http://<your_server_address>:<port>/gradio_api/run/<api_name>`
- Payload: `{"data": [arg1, arg2, ...]}`

---

## 6. Tips & Best Practices

- **Start Simple:** Get one function working with `gr.Interface` before moving to `gr.Blocks`.
- **Incremental Building:** Add components one by one and test.
- **Label Everything:** Use descriptive labels for all input/output components for clarity.
- **Use `gr.Examples`:** Provide sample inputs for users, especially for complex functions. This also helps you test.
- **Styling with `gr.themes`:** Easily change the look and feel (e.g., `theme=gr.themes.Soft()`).
- **Clear Docstrings & Type Hints:** Non-negotiable for MCP and good practice anyway.
- **Error Handling:** Your Python functions should handle potential errors gracefully. Gradio will display Python exceptions if not caught.

---

## State Management in `gr.Blocks`

### `gr.State()` for Session-Specific Data

- `gr.State()` is a special component used to store data that persists across multiple interactions within a single user's session. It's invisible in the UI.
- **Why Session-Specific?** Each user interacting with your Gradio app gets their own independent 'copy' of the `gr.State()` variable. This is crucial for applications like chatbots where each user has a separate conversation history, or for multi-step forms where each user's intermediate inputs are stored.
- **Initialization:** You can initialize `gr.State()` with a default value:  
  `chat_history_state = gr.State(value=[])`
- **Updating State:** To update the state, your event handler function must accept the current state as an input and return the new state. This new state is then passed back as an output to the `gr.State` component.

#### Example: Simple Counter

```python
import gradio as gr

with gr.Blocks() as demo:
    counter_state = gr.State(value=0)
    current_count_display = gr.Number(label="Current Count", value=0)
    increment_button = gr.Button("Increment")

    def increment_counter(current_val):
        new_val = current_val + 1
        # The order of return values must match the order of outputs
        return new_val, new_val # Update both state and display

    increment_button.click(
        fn=increment_counter,
        inputs=[counter_state],
        outputs=[counter_state, current_count_display]
    )
# demo.launch()
```

#### Example: Basic Chatbot History

```python
import gradio as gr

with gr.Blocks() as demo:
    chatbot_history_state = gr.State(value=[]) # List to store (user, bot) tuples
    user_input_textbox = gr.Textbox(label="Your Message")
    chat_display = gr.Chatbot(label="Conversation") # Gradio's Chatbot component

    def respond_to_user(message, history_list):
        # Simple echo bot for demonstration
        bot_response = f"You said: {message}"
        history_list.append((message, bot_response))
        # The Chatbot component expects a list of lists or list of tuples
        return history_list, "" # Return updated history and clear input textbox

    user_input_textbox.submit(
        fn=respond_to_user,
        inputs=[user_input_textbox, chatbot_history_state],
        outputs=[chat_display, user_input_textbox] # Update chat display and clear input
        # Note: We are not directly outputting to chatbot_history_state here,
        # but the 'history_list' variable IS the state.
        # To explicitly update gr.State, it must be in outputs.
        # A more explicit way to update state:
        # outputs=[chat_display, user_input_textbox, chatbot_history_state]
        # and the function would return: return history_list, "", history_list
    )
# demo.launch()
```

**Correction for Chatbot example:**  
To properly update `gr.State` and have it persist for the next call, it must be included in the outputs list of the event listener, and the function must return the new state value for that `gr.State` component.

```python
# Corrected Chatbot State Update Logic
# ... (inside respond_to_user function)
# return new_history_for_chatbot_component, new_value_for_textbox, new_history_for_state_component
# ...
# user_input_textbox.submit(
#     fn=respond_to_user,
#     inputs=[user_input_textbox, chatbot_history_state],
#     outputs=[chat_display, user_input_textbox, chatbot_history_state]
# )
```

**Use Cases:**  
Ideal for accumulating data (like chat logs, survey responses within a session), remembering user selections across steps in a wizard-like interface, or storing intermediate calculations that are needed for subsequent operations by the same user.

**Important:**  
`gr.State()` is in-memory and session-bound. It does not persist data if the user closes their browser tab, the server restarts, or across different users. For persistent storage, you'd need to integrate an external database or file system.

---

## Stateless Operations (Alternative to `gr.State()` for simple cases)

For many interactions, you might not need to explicitly manage persistent state with `gr.State()`. If an operation is purely functional (output depends only on current inputs) or if the necessary "state" is already present in visible UI components, you can often just pass data directly.

**How it works:**  
Data flows from input components, is processed by your Python function, and then updates output components. The "state" is effectively held by the current values of the visible components.

#### Example: A simple calculator

```python
import gradio as gr

def add_numbers(num1, num2):
    return num1 + num2

with gr.Blocks() as demo:
    num_input1 = gr.Number(label="First Number")
    num_input2 = gr.Number(label="Second Number")
    add_button = gr.Button("Add")
    result_output = gr.Number(label="Result")

    add_button.click(
        fn=add_numbers,
        inputs=[num_input1, num_input2],
        outputs=[result_output]
    )
# demo.launch()
```

**When to prefer this:**  
If the interaction doesn't require memory of past interactions within the same session (beyond what's visible on screen), this simpler approach is often cleaner and easier to reason about.

---

## Considerations for State Management

- **Resetting State:** Provide UI elements (e.g., a "Clear History" or "Reset Form" button) that call functions to reset your `gr.State()` variables to their initial values.
- **Complexity:** While `gr.State()` is powerful, heavily relying on many state variables can make your application logic more complex and harder to debug. Keep state management as simple as possible for your use case.
- **Data Volume:** Be mindful of storing very large amounts of data in `gr.State()`, as it's held in server memory for each active session. For extensive data, consider summarizing or offloading to a more robust storage solution if persistence is needed.
- **Consult Gradio Docs:** The official Gradio documentation ([gradio.app/docs](https://gradio.app/docs)) is excellent and comprehensive.

---

