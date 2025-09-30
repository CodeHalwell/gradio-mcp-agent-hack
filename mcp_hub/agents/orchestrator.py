"""Orchestrator Agent that coordinates all other agents."""
import asyncio
from typing import Dict, Any

from .question_enhancer import QuestionEnhancerAgent
from .web_search import WebSearchAgent
from .llm_processor import LLMProcessorAgent
from .citation_formatter import CitationFormatterAgent
from .code_generator import CodeGeneratorAgent
from .code_runner import CodeRunnerAgent

from ..config import api_config, model_config, app_config
from ..exceptions import APIError, ValidationError, CodeGenerationError
from ..logging_config import logger
from ..utils import make_llm_completion


class OrchestratorAgent:
    """
    Main orchestrator that coordinates all agents for the complete workflow.

    This agent manages the end-to-end workflow by coordinating question enhancement,
    web search, LLM processing, citation formatting, code generation, and code execution.
    It provides the primary interface for complex multi-step AI-assisted tasks.
    """
    
    def __init__(self):
        self.question_enhancer = QuestionEnhancerAgent()
        self.web_search = WebSearchAgent()
        self.llm_processor = LLMProcessorAgent()
        self.citation_formatter = CitationFormatterAgent()
        self.code_generator = CodeGeneratorAgent()
        self.code_runner = CodeRunnerAgent()
    
    def orchestrate(self, user_request: str) -> tuple[Dict[str, Any], str]:
        """
        Orchestrate the complete workflow: enhance question → search → generate code → execute.

        Manages the full AI-assisted workflow by coordinating all agents to provide
        comprehensive research, code generation, and execution. Returns both structured
        data and natural language summaries of the complete process.

        Args:
            user_request (str): The user's original request or question

        Returns:
            tuple[Dict[str, Any], str]: A tuple containing the complete result dictionary
                                       and a natural language summary of the process
        """
        try:
            logger.info(f"Starting orchestration for: {user_request[:100]}...")
            
            # Step 1: Enhance the question
            logger.info("Step 1: Enhancing question...")
            enhanced_result = self.question_enhancer.enhance_question(user_request, num_questions=3)
            sub_questions = enhanced_result.get('sub_questions', [user_request])
              # Step 2: Search for information
            logger.info("Step 2: Searching for information...")
            search_results = []            
            search_summaries = []
            
            for i, question in enumerate(sub_questions[:2]):  # Limit to 2 questions to avoid too many searches
                logger.info(f"Processing question {i+1}: {question}")
                try:
                    search_result = self.web_search.search(question)
                    logger.info(f"Search result for question {i+1}: {search_result}")

                    # Extract results and summary regardless of status key
                    results = search_result.get('results', [])
                    summary = search_result.get('tavily_answer', search_result.get('summary', ''))

                    if results or summary:  # Treat as success if any results or summary found
                        logger.info(f"Question {i+1} - Found {len(results)} results")
                        logger.info(f"Question {i+1} - Summary: {summary[:100]}...")

                        # Add to collections
                        search_results.extend(results)
                        search_summaries.append(summary)

                        logger.info(f"Question {i+1} - Successfully added {len(results)} results to collection")
                        logger.info(f"Question {i+1} - Current total search_results: {len(search_results)}")
                        logger.info(f"Question {i+1} - Current total search_summaries: {len(search_summaries)}")
                    else:
                        error_msg = search_result.get('error', 'Unknown error or no results returned') 
                        logger.warning(f"Search failed for question {i+1}: {error_msg}")

                except Exception as e:
                    logger.error(f"Exception during search for question '{question}': {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
            
            logger.info(f"Total search results collected: {len(search_results)}")
            logger.info(f"Total search summaries: {len(search_summaries)}")
            for i, result in enumerate(search_results[:3]):
                logger.info(f"Search result {i+1}: {result.get('title', 'No title')[:50]}...")
            
            # Step 3: Create grounded context
            logger.info("Step 3: Creating grounded context...")
            grounded_context = ""
            if search_results:
                # Combine search results into context
                context_parts = []
                for result in search_results[:5]:  # Limit to top 5 results
                    context_parts.append(f"Title: {result.get('title', 'N/A')}")
                    context_parts.append(f"Content: {result.get('content', 'N/A')}")
                    context_parts.append(f"URL: {result.get('url', 'N/A')}")
                    context_parts.append("---")
                
                grounded_context = "\n".join(context_parts)
            
            # If no search results, use a generic context
            if not grounded_context:
                grounded_context = f"User request: {user_request}\nNo additional web search context available."
              # Step 4: Generate code
            logger.info("Step 4: Generating code...")
            logger.info(f"Grounded context length: {len(grounded_context)}")
            code_result, code_summary = self.code_generator.generate_code(user_request, grounded_context)
            logger.info(f"Code generation result: {code_result}")
            logger.info(f"Code generation summary: {code_summary[:200]}...")
            
            code_string = ""
            if code_result.get('status') == 'success':
                # Use raw_output (string) for display, generated_code (compiled) for execution
                code_string = code_summary  # This is the raw string output
                logger.info(f"Successfully extracted code_string with length: {len(code_string)}")
                logger.info(f"Code preview: {code_string[:200]}...")
            else:
                logger.warning(f"Code generation failed: {code_result.get('error', 'Unknown error')}")
            
            # Step 5: Execute code if available
            execution_output = ""
            if code_string:
                logger.info("Step 5: Executing code...")
                try:
                    # Use async execution for better performance
                    import asyncio
                    execution_output = asyncio.run(self.code_runner.run_code_async(code_string))
                except Exception as e:
                    execution_output = f"Execution failed: {str(e)}"
                    logger.warning(f"Code execution failed: {e}")
            
            # Step 6: Format citations
            logger.info("Step 6: Formatting citations...")
            citations = []
            for result in search_results:
                if result.get('url'):
                    citations.append(f"{result.get('title', 'Untitled')} - {result.get('url')}")
              # Compile final result
            logger.info("=== PRE-FINAL RESULT DEBUG ===")
            logger.info(f"search_results length: {len(search_results)}")
            logger.info(f"search_summaries length: {len(search_summaries)}")
            logger.info(f"code_string length: {len(code_string)}")
            logger.info(f"execution_output length: {len(execution_output)}")
            logger.info(f"citations length: {len(citations)}")
            

            logger.info("=== GENERATING EXECUTIVE SUMMARY ===")
            # Sample first search result
            if search_results:
                logger.info(f"First search result: {search_results[0]}")

            prompt = f"""
            The user asked about {user_request} which yielded this summary: {search_summaries} 
            
            During the orchestration, you generated the following code: {code_string}

            The code was executed in a secure sandbox environment, and the output was <executed_code>{execution_output}</executed_code>.

            If there was no output in the executed_code tags, please state how to answer the user's request showing the code required.
            State that the code you are giving them has not been executed, and that they should run it in their own environment.

            Please provide a short and concise summary of the code that you wrote, including the user request, the summaries provided and the code generated.
            Explain how the code addresses the user's request, what it does, and any important details about its execution.

            Touch upon the other methods available that were found in the search results, and how they relate to the user's request.
            
            Please return the result in natural language only, without any code blocks, unless as stated above, there was no code executed in the sandbox and then you should give them the code
            as a code block.
            References to code can be made to explain why particular code has been used regardless of sandbox execution, e.g. discuss why the LinerRegression module was used  from scikit-learn etc.
            
            If no code was generated, apologise, please state that clearly the code generation failed in the sandbox, this could be due to restriction
            or the code being too complex for the sandbox to handle.

            Note, if appropriate, indicate how the code can be modified to include human input etc. as this is a banned keyword in the sandbox.

            The response should be directed at the user, in a friendly and helpful manner, as if you were a human assistant helping the user with their request.

            **Summary Requirements:**

            - The summary should be concise, no more than 500 words.
            - It should clearly explain how the code addresses the user's request.
            - It should only include code if there was no execution output, and then it should be in a code block. (if there is executed_code, this will be returned by
            another process and therefor you dont need to do it here)
            - The summary should be written in a friendly and helpful tone, as if you were a human assistant helping the user with their request.

            """

            messages = [{"role": "user", 
                         "content": prompt}]
            
            logger.info(f"LLM provider is: {api_config.llm_provider}, model used: {model_config.get_model_for_provider('llm_processor', api_config.llm_provider)}")
            # Last call to LLM to summarize the entire orchestration
            overall_summary = make_llm_completion(
                model=model_config.get_model_for_provider("llm_processor", api_config.llm_provider),
                messages=messages,
                temperature=app_config.llm_temperature
            )            
            logger.info("Overall summary generated:")
            
            final_result = {
                "status": "success",
                "user_request": user_request,
                "sub_questions": sub_questions,
                "search_results": search_results[:5],
                "search_summaries": search_summaries,
                "code_string": code_string,
                "execution_output": execution_output,
                "citations": citations,
                "final_summary": f"{overall_summary}",
                "message": "Orchestration completed successfully"
            }
            
            # Create clean summary for display
            final_narrative = f"## 🎯 Request: {user_request}\n\n{overall_summary}"
            
            logger.info("Orchestration completed successfully")
            return final_result, final_narrative
            
        except (ValidationError, APIError, CodeGenerationError) as e:
            logger.error(f"Orchestration failed: {str(e)}")
            # Create execution log for error case
            execution_log = f"Error during orchestration: {str(e)}"
            return {"error": str(e), "execution_log": execution_log}, str(e)
        except Exception as e:
            logger.error(f"Unexpected error in orchestration: {str(e)}")
            # Create execution log for error case
            execution_log = f"Unexpected error: {str(e)}"
            return {"error": f"Unexpected error: {str(e)}", "execution_log": execution_log}, str(e)
    
    def _format_search_results(self, results):
        """Format search results into a combined text snippet."""
        formatted_parts = []
        for result in results:
            title = result.get('title', 'No title')
            content = result.get('content', 'No content')
            url = result.get('url', 'No URL')
            formatted_parts.append(f"Title: {title}\nContent: {content}\nURL: {url}\n---")
        
        return "\n".join(formatted_parts)
    
    async def _run_subquestion_async(self, sub_question: str, user_request: str) -> tuple:
        """Process a single sub-question asynchronously."""
        try:
            # Search
            search_result = await self.web_search.search_async(sub_question)
            if search_result.get("error"):
                logger.warning(f"Async search failed for sub-question: {search_result['error']}")
                return None, None
            
            # Format search results
            results = search_result.get("results", [])[:app_config.max_search_results]
            formatted_text = self._format_search_results(results)
            
            # Process search results
            llm_summary = await self.llm_processor.async_process(
                formatted_text, 
                "summarize", 
                f"Context of user request: {user_request}"
            )
            
            # Prepare result
            result_data = {
                "status": "success",
                "sub_question": sub_question,
                "user_request": user_request,
                "search_results": results,
                "search_summary": llm_summary.get('llm_processed_output', '')
            }
            
            # Create summary parts
            summary_parts = []
            summary_parts.append(f"## Subquestion: {sub_question}")
            summary_parts.append("### Research Summary:")
            summary_parts.append(llm_summary.get('llm_processed_output', 'No summary available'))
            
            # Add sources if available
            citations = []
            for result in results:
                if result.get('url'):
                    citations.append(f"{result.get('title', 'Untitled')} - {result.get('url')}")
            
            if citations:
                summary_parts.append("### Sources:")
                for i, citation in enumerate(citations, 1):
                    summary_parts.append(f"{i}. {citation}")
            
            clean_summary = "\n\n".join(summary_parts)
            
            logger.info("Subquestion processing completed successfully")
            return result_data, clean_summary
            
        except Exception as e:
            logger.error(f"Subquestion processing failed: {e}")
            error_result = {
                "status": "error",
                "user_request": user_request,
                "sub_question": sub_question,
                "error": str(e),
                "message": "Subquestion processing failed"
            }
            return error_result, f"❌ Error: {str(e)}"
