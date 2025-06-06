import os
import json
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient
import modal
import textwrap
import base64
import marshal
import types

# ----------------------------------------
# Load .env variables
# ----------------------------------------
load_dotenv()  # Populate os.environ from your .env file

# ----------------------------------------
# Nebius (OpenAI-compatible) Setup
# ----------------------------------------
NEBIUS_API_KEY = os.environ.get("NEBIUS_API_KEY", "")
if not NEBIUS_API_KEY:
    raise RuntimeError("Please set NEBIUS_API_KEY in your .env file.")

nebius_client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=NEBIUS_API_KEY,
)

# ----------------------------------------
# Modal Sandbox–based Code Runner
# ----------------------------------------
# Look up (or create) your Modal App at module scope
app = modal.App.lookup("my-sandbox-app", create_if_missing=True)

def code_runner(code_or_obj, *, app) -> str:
    if isinstance(code_or_obj, str):
        payload = code_or_obj

    elif isinstance(code_or_obj, types.CodeType):
        b64 = base64.b64encode(marshal.dumps(code_or_obj)).decode()
        payload = textwrap.dedent(f"""
            import base64, marshal, types, traceback
            code = marshal.loads(base64.b64decode({b64!r}))
            try:
                exec(code, {{'__name__': '__main__'}})
            except Exception:
                traceback.print_exc()
        """).lstrip()     # remove leading blank line too

    else:
        return "Error: input must be str or types.CodeType"

    try:
        sb = modal.Sandbox.create(app=app)
        proc = sb.exec("python", "-c", payload)
        output = proc.stdout.read() + proc.stderr.read()
        sb.terminate()
        return output
    except Exception as e:
        try:
            sb.terminate()
        except Exception:
            pass
        return f"Error executing code in Modal sandbox: {e}"

    
# ----------------------------------------
# Agent: Question Enhancer (splits into 3 sub-questions)
# ----------------------------------------
def agent_question_enhancer(user_request: str) -> dict:
    """
    Splits a single user query into three distinct sub-questions.
    Uses Qwen/Qwen3-32B-fast to produce structured JSON output.
    """
    if not user_request or not user_request.strip():
        return {"error": "User request cannot be empty.", "sub_questions": []}

    prompt_text = f"""
        You are an AI assistant that must break a single user query into three distinct, non-overlapping sub-questions.
        Each sub-question should explore a different technical angle of the original request.
        Output must be valid JSON with a top-level key "sub_questions" whose value is an array of strings—no extra keys, no extra prose.

        User Request: "{user_request}"

        Respond with exactly:
        {{
        "sub_questions": [
            "First enhanced sub-question …",
            "Second enhanced sub-question …",
            "Third enhanced sub-question …"
        ]
        }}
        (We asked for three, but we'll accept however many you actually give us, as long as it’s a JSON array of strings under "sub_questions".)
        """

    try:
        completion = nebius_client.chat.completions.create(
            model="Qwen/Qwen3-4B-fast",
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.0,
            response_format={
                "type": "json_object",
                "object": {
                    "sub_questions": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
            },
        )
        resp_str = completion.to_json()
        resp_obj = json.loads(resp_str)
        choices = resp_obj.get("choices", [])
        if not (choices and isinstance(choices, list)):
            return {"error": "Unexpected response structure from Qwen.", "sub_questions": []}
        raw_output = choices[0].get("message", {}).get("content", "").strip()
    except Exception as e:
        return {"error": f"Nebius API Error in question enhancer: {str(e)}", "sub_questions": []}

    # Strip ``` fences if present
    text = raw_output
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1].strip()
        else:
            text = text.strip("```").strip()

    # Extract JSON object
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        return {"error": "Failed to locate JSON object in model output.", "sub_questions": []}

    json_candidate = text[start_idx : end_idx + 1]
    try:
        parsed = json.loads(json_candidate)
    except json.JSONDecodeError as je:
        return {"error": f"Failed to parse sub-questions JSON: {str(je)}", "sub_questions": []}

    if not isinstance(parsed, dict) or "sub_questions" not in parsed:
        return {"error": "JSON does not contain a 'sub_questions' key.", "sub_questions": []}

    sub_qs = parsed.get("sub_questions")
    if not isinstance(sub_qs, list) or not all(isinstance(q, str) for q in sub_qs):
        return {"error": "Expected 'sub_questions' to be a list of strings.", "sub_questions": []}

    return {"sub_questions": sub_qs}

# ----------------------------------------
# Agent: Web Search (reads TAVILY_API_KEY from environment)
# ----------------------------------------
def agent_web_search(query: str) -> dict:
    """
    Perform a web search using the Tavily API, returning up to 3 results.
    """
    tavily_api_key = os.environ.get("TAVILY_API_KEY", "")
    if not TavilyClient:
        return {
            "error": "TavilyClient is not installed. Run: pip install tavily-python",
            "query": query,
            "results": [],
        }
    if not tavily_api_key or not tavily_api_key.startswith("tvly-"):
        return {
            "error": "A valid TAVILY_API_KEY is required in your .env file.",
            "query": query,
            "results": [],
        }
    if not isinstance(query, str) or not query.strip():
        return {"error": "Query must be a non-empty string.", "query": query, "results": []}

    try:
        client = TavilyClient(api_key=tavily_api_key)
        response = client.search(
            query=query, search_depth="basic", max_results=3, include_answer=True
        )
        return {
            "query": response.get("query", query),
            "tavily_answer": response.get("answer"),
            "results": response.get("results", []),
            "data_source": "Tavily Search API",
        }
    except Exception as e:
        return {"error": f"Tavily API Error: {str(e)}", "query": query, "results": []}

# ----------------------------------------
# Agent: LLM Processor (Nebius)
# ----------------------------------------
def agent_llm_processor(text_input: str, task: str, context: str = None) -> dict:
    """
    Use Nebius (Meta-Llama-3.1-8B-Instruct) for 'summarize', 'reason', or 'extract_keywords'.
    Extracts the assistant’s reply from the returned JSON.
    """
    if not text_input or not text_input.strip():
        return {"error": "Input text cannot be empty.", "task": task, "processed_output": None}
    if not task or not task.strip():
        return {"error": "Task cannot be empty.", "task": task, "processed_output": None}

    t_lower = task.lower()
    if t_lower == "reason":
        prompt_text = f"""Analyze this text and provide detailed reasoning:

{text_input}
"""
        if context:
            prompt_text += f"\n\nAdditional context: {context}"
        prompt_text += "\n\nReasoning:"
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    elif t_lower == "summarize":
        prompt_text = f"""Summarize concisely (100–150 words):

{text_input}
"""
        if context:
            prompt_text += f"\n\nKeep in mind this context: {context}"
        prompt_text += "\n\nSummary:"
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    elif t_lower == "extract_keywords":
        prompt_text = f"""Extract key terms/entities (comma-separated) from:

{text_input}
"""
        if context:
            prompt_text += f"\n\nFocus on this context: {context}"
        prompt_text += "\n\nKeywords:"
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    else:
        return {
            "error": f"Unsupported LLM task: {task}. Choose 'summarize', 'reason', or 'extract_keywords'.",
            "input_text": text_input,
            "processed_output": None,
        }

    try:
        completion = nebius_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.6,
        )
        resp_str = completion.to_json()
        resp_obj = json.loads(resp_str)

        choices = resp_obj.get("choices", [])
        if choices and isinstance(choices, list):
            first_choice = choices[0]
            message = first_choice.get("message", {})
            output_text = message.get("content", "").strip()
        else:
            output_text = "Error: Unexpected Nebius response structure."
    except Exception as e:
        return {"error": f"Nebius API Error: {str(e)}", "input_text": text_input, "processed_output": None}

    return {
        "input_text": text_input,
        "task": task,
        "provided_context": context,
        "llm_processed_output": output_text,
        "llm_model_used": model_name,
    }

# ----------------------------------------
# Agent: Citation Formatter
# ----------------------------------------
def agent_citation_formatter(text_block: str) -> dict:
    """
    Given a block of text (with URLs), extracts URLs and produces simple APA-style citations.
    """
    import re

    if not isinstance(text_block, str) or not text_block.strip():
        return {"error": "Text block is empty.", "formatted_citations": []}

    urls = re.findall(r"(https?://[^\s]+)", text_block)
    if not urls:
        return {"error": "No URLs found to cite.", "formatted_citations": []}

    citations = []
    for u in urls:
        try:
            domain = u.split("/")[2]
            title = domain.replace("www.", "").split(".")[0].capitalize()
            year = str(os.environ.get("CURRENT_YEAR", "2025"))
            citation = f"{title}. ({year}). Retrieved from {u}"
            citations.append(citation)
        except Exception:
            continue

    if not citations:
        return {"error": "Failed to format any citations.", "formatted_citations": []}

    return {"formatted_citations": citations, "error": None}


# ----------------------------------------
# Agent: Code Generator
# ----------------------------------------
def agent_code_generator(user_request: str, grounded_context: str, *, max_attempts: int = 3) -> dict:
    """
    Generate a valid Python code snippet based on the user request
    and grounded context (summaries + citations). The function checks
    that the returned snippet compiles before returning it.
    """
    if not user_request or not user_request.strip():
        return {"error": "User request cannot be empty.", "generated_code": ""}

    system_prompt = f"""
            You are an expert Python developer. Given the user’s request and the following
            grounded context (search summaries and citations), generate a Python code
            snippet that directly addresses the user’s needs. Ensure the code is valid,
            complete, and runnable.

            User Request:
            \"\"\"{user_request}\"\"\"

            Grounded Context:
            \"\"\"{grounded_context}\"\"\"

            Provide only the Python code and never under any circumstance include any
            explanations in your response. **Do not include back ticks or the word python and dont include input fields**

            for example,

            import requests
            response = requests.get("https://api.example.com/data")
            print(response.json())

            or

            def add_numbers(a, b):
                return a + b
            result = add_numbers(5, 10)
            print(result)

            NEVER include input()
        """

    for attempt in range(1, max_attempts + 1):
        try:
            completion = nebius_client.chat.completions.create(
                model="Qwen/Qwen2.5-Coder-32B-Instruct-fast",
                messages=[{"role": "user", "content": system_prompt}],
                temperature=0.2,
            )
            raw_output = completion.choices[0].message.content.strip()
        except Exception as e:
            return {"error": f"Nebius API Error in code generator: {str(e)}", "generated_code": ""}
        
        print(raw_output)
        
        code_compiled = compile(raw_output, "<string>", "exec")

        return {"generated_code": code_compiled}, raw_output

    return {"error": "Failed to generate valid Python after multiple attempts.", "generated_code": ""}

# ----------------------------------------
# Agent: Orchestrator (3-question code workflow)
# ----------------------------------------
def agent_orchestrator(user_request: str) -> dict:
    """
    Orchestrator for new workflow:
      1) Enhance question → 3 sub-questions
      2) For each sub-question:
         a) web_search → top-3 results
         b) llm_processor:summarize → 100–150 word summary
         c) citation_formatter → citations from URLs
      3) Combine sub-summaries into one grounded context
      4) Generate Python code snippet using grounded context
      5) Run code in Modal sandbox
      6) Return code, explanation, sources, and execution outcome
      7) Return the final summary in natural language, with generated code and citations.

    Returns:
      {
        "user_request": str,
        "final_summary": str,
        "generated_code": str,
        "code_output": str,
        "citations": [ ... ],
        "citation_error": str or None,
        "execution_log": [ ... ]
      }
    """
    execution_log = []

    if not user_request or not user_request.strip():
        return {"error": "User request cannot be empty.", "execution_log": execution_log}

    # Step 1: Enhance into three sub-questions
    step0 = {"step": 0, "tool": "question_enhancer", "input": user_request}
    enhancer_out = agent_question_enhancer(user_request)
    step0["result"] = enhancer_out
    execution_log.append(step0)
    if enhancer_out.get("error"):
        return {"error": f"Question enhancement failed: {enhancer_out['error']}", "execution_log": execution_log}

    sub_questions = enhancer_out.get("sub_questions", [])
    if not sub_questions:
        return {"error": "No sub-questions returned.", "execution_log": execution_log}

    # Step 2: For each sub-question, perform search, summarization, citation
    all_sub_summaries = []
    all_citations = []
    citation_errors = []

    for idx, sub_q in enumerate(sub_questions, start=1):
        # 2a. Web search
        step_search = {"step": f"subquestion_{idx}_web_search", "tool": "web_search", "query": sub_q}
        search_res = agent_web_search(sub_q)
        step_search["result"] = search_res
        execution_log.append(step_search)
        if search_res.get("error"):
            return {"error": f"Web search failed for sub-question {idx}: {search_res['error']}", "execution_log": execution_log}

        # Build snippet from top 3 results
        snippets = []
        for res_idx, item in enumerate(search_res.get("results", [])[:3]):
            title = item.get("title", "No Title")
            url = item.get("url", "")
            content_snippet = item.get("content", "")
            snippets.append(f"Result {res_idx+1}:\nTitle: {title}\nURL: {url}\nSnippet: {content_snippet}\n")
        combined_snippet = "\n".join(snippets).strip()

        # 2b. Summarization
        step_summarize = {
            "step": f"subquestion_{idx}_summarize",
            "tool": "llm_processor",
            "task": "summarize",
            "input_source": f"subquestion_{idx}_web_search_snippets",
            "input": combined_snippet
        }
        llm_out = agent_llm_processor(combined_snippet, "summarize", context=None)
        step_summarize["result"] = llm_out
        execution_log.append(step_summarize)
        if llm_out.get("error"):
            return {"error": f"LLM summarization failed for sub-question {idx}: {llm_out['error']}", "execution_log": execution_log}

        sub_summary = llm_out.get("llm_processed_output", "")
        all_sub_summaries.append(sub_summary)

        # 2c. Citation formatting
        step_cite = {
            "step": f"subquestion_{idx}_citation_formatter",
            "tool": "citation_formatter",
            "input_source": f"subquestion_{idx}_web_search_snippets",
            "input": combined_snippet
        }
        citation_out = agent_citation_formatter(combined_snippet)
        step_cite["result"] = citation_out
        execution_log.append(step_cite)
        if citation_out.get("error"):
            citation_errors.append(f"Sub-question {idx}: {citation_out['error']}")
        else:
            all_citations.extend(citation_out.get("formatted_citations", []))

    # Step 3: Combine sub-summaries into grounded context
    combine_input = "\n\n".join([f"Summary {i+1}:\n{sub}" for i, sub in enumerate(all_sub_summaries)])
    step_combine = {
        "step": 3,
        "tool": "llm_processor",
        "task": "summarize",
        "input_source": "sub_summaries",
        "input": combine_input
    }
    combine_out = agent_llm_processor(
        combine_input,
        "summarize",
        context="Combine these sub-summaries into a single cohesive context for code generation."
    )
    step_combine["result"] = combine_out
    execution_log.append(step_combine)
    if combine_out.get("error"):
        return {"error": f"Failed to combine summaries: {combine_out['error']}", "execution_log": execution_log}

    final_summary = combine_out.get("llm_processed_output", "")

    # Step 4: Generate Python code snippet
    grounded_context = final_summary + "\n\nCitations:\n" + "\n".join(all_citations)
    step_code_gen = {
        "step": 4,
        "tool": "code_generator",
        "input": {
            "user_request": user_request,
            "grounded_context": grounded_context
        }
    }
    code_out, code_string = agent_code_generator(user_request, grounded_context)
    step_code_gen["result"] = code_out
    execution_log.append(step_code_gen)
    if code_out.get("error"):
        return {"error": f"Code generation failed: {code_out['error']}", "execution_log": execution_log}

    generated_code = code_out.get("generated_code", "")

    # Step 5: Run generated code in Modal sandbox
    step_run = {"step": 5, "tool": "code_runner", "input": generated_code}
    code_output = code_runner(generated_code, app=app)
    step_run["result"] = {"code_string": code_string,"code_output": code_output}
    execution_log.append(step_run)

    citation_error = None
    if citation_errors:
        citation_error = "; ".join(citation_errors)

    # Step 7 Return the final summary in natural language, with generated code and citations using the LLM to summarise the entire process
    final_summary = nebius_client.chat.completions.create(
        model="Qwen/Qwen3-32B-fast",
        messages=[
            {
                "role": "user",
                "content": f"""
                    Summarize the entire research and code generation process:
                    User Request: {user_request}
                    Final Summary: {final_summary}
                    Generated Code: {generated_code}
                    Code Output: {code_output}
                    Citations: {', '.join(all_citations)}
                    
                    Provide a concise summary of the entire process, including the user request, final summary, generated code, execution output, and citations.
                """
            }
        ],
        temperature=0.5,
    ).choices[0].message.content.strip()
    
    
    

    return {
        "user_request": user_request,
        "final_summary": final_summary,
        "code_string": code_string,
        "generated_code": generated_code,
        "code_output": code_output,
        "citations": all_citations,
        "citation_error": citation_error,
        "execution_log": execution_log
    }, final_summary

# ----------------------------------------
# Gradio UI / MCP Server Setup
# ----------------------------------------
with gr.Blocks(title="Deep Research & Code Assistant Hub", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        ## Deep Research & Code Assistant Hub

        **Workflow**:
        1. Break the user’s request into three enhanced sub-questions.
        2. Perform Tavily web search (top-3 results) for each sub-question.
        3. Summarize each set of snippets via Nebius LLM.
        4. Extract APA-style citations from each snippet block.
        5. Combine summaries into a grounded context.
        6. Generate a Python code snippet based on that context.
        7. Run the generated code in a Modal sandbox.
        8. Return the final summary, generated code, execution output, and citations.
        """
    )

    with gr.Tab("Orchestrator (Research → Code Workflow)"):
        gr.Interface(
            fn=agent_orchestrator,
            inputs=[
                gr.Textbox(
                    label="Your High-Level Request",
                    lines=3,
                    placeholder="E.g. 'Write Python code to scrape the latest stock prices and plot a graph.'"
                )
            ],
            outputs=gr.JSON(label="Orchestrated Code Output"),
            title="AI Research & Code Assistant",
            description=(
                "1) Splits into 3 sub-questions → "
                "2) Tavily search & summarization per sub-question → "
                "3) Combine into context → "
                "4) Generate Python code → "
                "5) Execute code via Modal → "
                "6) Return code, output, and citations."
            ),
            api_name="agent_orchestrator_service",
        )

    with gr.Tab("Agent: Question Enhancer"):
        gr.Interface(
            fn=agent_question_enhancer,
            inputs=[
                gr.Textbox(
                    label="Original User Request",
                    lines=2,
                    placeholder="Enter your question to be split into 3 sub-questions…"
                )
            ],
            outputs=gr.JSON(label="Enhanced Sub-Questions"),
            title="Question Enhancer Agent",
            description="Splits a single user query into 3 distinct sub-questions using Qwen/Qwen3-32B-fast.",
            api_name="agent_question_enhancer_service",
        )

    with gr.Tab("Agent: Web Search"):
        gr.Interface(
            fn=agent_web_search,
            inputs=[gr.Textbox(label="Search Query", placeholder="Enter search term…")],
            outputs=gr.JSON(label="Web Search Results (Tavily)"),
            title="Web Search Agent",
            description="Perform a Tavily web search (top-3 results).",
            api_name="agent_web_search_service",
        )

    with gr.Tab("Agent: LLM Processor (Nebius)"):
        gr.Interface(
            fn=agent_llm_processor,
            inputs=[
                gr.Textbox(label="Text to Process", lines=5, placeholder="Enter text for the LLM…"),
                gr.Dropdown(
                    choices=["summarize", "reason", "extract_keywords"],
                    value="summarize",
                    label="LLM Task",
                ),
                gr.Textbox(label="Optional Context", lines=2, placeholder="Background info…"),
            ],
            outputs=gr.JSON(label="LLM Processed Output"),
            title="LLM Processing Agent",
            description="Use Meta-Llama-3.1-8B-Instruct to summarize, reason, or extract keywords.",
            api_name="agent_llm_processor_service",
        )

    with gr.Tab("Agent: Citation Formatter"):
        gr.Interface(
            fn=agent_citation_formatter,
            inputs=[
                gr.Textbox(
                    label="Text Block (with URLs)",
                    lines=5,
                    placeholder="Paste text containing URLs to generate APA citations…"
                )
            ],
            outputs=gr.JSON(label="Formatted APA-style Citations"),
            title="Citation Formatter Agent",
            description="Extracts URLs from text and returns a list of APA-style citations.",
            api_name="agent_citation_formatter_service",
        )

    with gr.Tab("Agent: Code Generator"):
        gr.Interface(
            fn=agent_code_generator,
            inputs=[
                gr.Textbox(label="Original Request", lines=2, placeholder="Enter your high-level request…"),
                gr.Textbox(label="Grounded Context", lines=8, placeholder="Paste combined summaries + citations…")
            ],
            outputs=gr.JSON(label="Generated Python Code"),
            title="Code Generator Agent",
            description="Generates a Python code snippet based on the grounded context using Meta-Llama-3.1-8B-Instruct.",
            api_name="agent_code_generator_service",
        )

    with gr.Tab("Agent: Code Runner (Modal)"):
        gr.Interface(
            fn=code_runner,
            inputs=[gr.Code(label="Python Code to Execute", language="python")],
            outputs=[gr.Textbox(label="Execution Output")],
            title="Code Runner Agent",
            description="Executes the provided Python code in a Modal sandbox and returns the output.",
            api_name="agent_code_runner_service",
        )

if __name__ == "__main__":
    print("Launching Deep Research & Code Assistant Hub…")
    print("Ensure your `.env` has:\n  TAVILY_API_KEY=tvly-...\n  NEBIUS_API_KEY=nb-...")
    print("Install dependencies: pip install gradio requests tavily-python openai python-dotenv modal")
    demo.launch(mcp_server=True, server_name="127.0.0.1")
    print("\nMCP schema available at http://127.0.0.1:7860/gradio_api/mcp/schema")
