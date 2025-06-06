import gradio as gr

# 1. Define your Python function
def greet(name):
    return "Hello, " + name + "!"

# 2. Create a Gradio Interface
iface = gr.Interface(
    fn=greet,
    inputs=gr.Textbox(label="Enter your name"),
    outputs=gr.Textbox(label="Greeting")
)

# 3. Launch the Interface
if __name__ == "__main__":
    iface.launch()