from gradio_client import Client

def print_human_readable_result(result):
    # Print main request and status
    if isinstance(result, tuple):
        result = next((item for item in result if isinstance(item, dict)), result[0])
    print("Status:", result.get('status', 'N/A'))
    print("Status:", result.get('status', 'N/A'))
    print("User Request:", result.get('user_request', 'N/A'))
    print("\nSub-Questions:")
    for i, sub_q in enumerate(result.get('sub_questions', []), 1):
        print(f"  {i}. {sub_q}")

    print("\nSearch Summaries:")
    for i, summary in enumerate(result.get('search_summaries', []), 1):
        print(f"  {i}. {summary}")

    print("\nSearch Results:")
    for i, res in enumerate(result.get('search_results', []), 1):
        print(f"  {i}. {res['title']}\n     URL: {res['url']}\n     Content: {res['content'][:100]}{'...' if len(res['content']) > 100 else ''}\n     Score: {res['score']:.3f}")

    print("\nGenerated Code:\n" + result.get('code_string', 'N/A'))

    print("\nExecution Output:\n" + result.get('execution_output', 'N/A'))

    print("\nCitations:")
    for i, cit in enumerate(result.get('citations', []), 1):
        print(f"  {i}. {cit}")

    print("\nFinal Summary:\n" + result.get('final_summary', 'N/A'))

    print("\nOrchestration Message:", result.get('message', 'N/A'))

client = Client("http://127.0.0.1:7860/")
result = client.predict(
		user_request="How do I calculate the sum of an array in Python?",
		api_name="/process_orchestrator_request"
)
print_human_readable_result(result)