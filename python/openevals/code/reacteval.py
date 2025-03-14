"""
React code evaluator that renders React components and evaluates them using screenshots.

This module provides functions for creating evaluators that assess React code
by rendering it in a headless browser, taking screenshots, and using an LLM to evaluate
whether the visual output matches a provided description.
"""

import asyncio
import os
import tempfile
import traceback
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from openevals.code.base import (
    _create_async_base_code_evaluator,
    _create_base_code_evaluator,
)
from openevals.types import EvaluatorResult

from openevals.llm import create_llm_as_judge, create_async_llm_as_judge


# React evaluation prompt
REACT_EVAL_PROMPT = """
You are an expert React developer and UI evaluator responsible for determining if a rendered React component 
matches a given description. You will be provided with:

1. A description of what the React component should look like and how it should function
2. A screenshot of the rendered React component

Your task is to evaluate whether the rendered component accurately matches the provided description.
Be detailed and specific about what aspects match or don't match. Consider visual elements, layout,
functionality described, and overall adherence to the description.

Description:
{description}

Evaluation criteria:
- Does the component visually match what was described?
- Are all the described elements present?
- Does the layout match what was described?
- Would the component function as described (even if you can only see a static image)?

Score with TRUE if the component matches the description well, or FALSE if it has significant issues or mismatches.
"""


def _prepare_react_environment(code: str) -> str:
    """
    Prepares a full HTML file with the necessary React environment to render the component.
    
    Args:
        code: The React component code to render.
        
    Returns:
        The full HTML document as a string.
    """
    # Create a minimal React environment with CDN imports
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8" />
        <title>React Component Preview</title>
        <script>
            // Signal when page is fully loaded
            window.onload = function() {{
                document.body.setAttribute('data-loaded', true);
            }};
            
            // Detect errors in script loading
            window.addEventListener('error', function(e) {{
                console.error('Script error:', e);
                document.body.setAttribute('data-error', e.message);
            }}, true);
        </script>
        <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
        <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
        <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
        <style>
            body {{
                font-family: sans-serif;
                padding: 20px;
            }}
            #root {{
                border: 1px solid #eee;
                padding: 20px;
                border-radius: 8px;
            }}
            #render-status {{
                display: none;
                padding: 10px;
                margin-top: 10px;
                border-radius: 4px;
            }}
            #render-status.success {{
                display: block;
                background-color: #e6ffe6;
                color: #006600;
            }}
            #render-status.error {{
                display: block;
                background-color: #ffe6e6;
                color: #cc0000;
            }}
        </style>
    </head>
    <body>
        <div id="root"></div>
        <div id="render-status"></div>

        <script type="text/babel">
        {code}
        
        // Signal when React rendering is complete
        function signalRenderComplete(success, message) {{
            const statusEl = document.getElementById('render-status');
            statusEl.textContent = message;
            statusEl.className = success ? 'success' : 'error';
            document.body.setAttribute('data-react-rendered', success ? true : false);
        }}
        
        // Try to find and render the component
        try {{
            // Wait for Babel to be ready
            setTimeout(() => {{
                try {{
                    const componentNames = Object.keys(window).filter(
                        key => typeof window[key] === 'function' && 
                        /^[A-Z]/.test(key) && 
                        key !== 'React' && 
                        key !== 'ReactDOM'
                    );
    
                    // If a component is found in the global scope, render it
                    if (componentNames.length > 0) {{
                        const componentName = componentNames[0];
                        const rootElement = document.getElementById('root');
                        
                        // Use createRoot API for React 18
                        const root = ReactDOM.createRoot(rootElement);
                        root.render(React.createElement(window[componentName]));
                        
                        signalRenderComplete(Boolean(true), "Component '" + componentName + "' rendered successfully");
                    }} else {{
                        console.error('No valid component found in the code.');
                        document.getElementById('root').innerHTML = '<div style="color: red;">No valid component found.</div>';
                        signalRenderComplete(Boolean(false), 'No valid React component found');
                    }}
                }} catch (err) {{
                    console.error('Error in delayed rendering:', err);
                    document.getElementById('root').innerHTML = '<div style="color: red;">Error: ' + err.message + '</div>';
                    signalRenderComplete(Boolean(false), 'Error: ' + err.message);
                }}
            }}, 500); // Short delay to ensure Babel has processed the code
        }} catch (e) {{
            console.error('Error rendering component:', e);
            document.getElementById('root').innerHTML = '<div style="color: red;">Error: ' + e.message + '</div>';
            signalRenderComplete(Boolean(false), 'Error: ' + e.message);
        }}
        </script>
    </body>
    </html>
    """
    return html_template


async def _evaluate_react_code_with_llm_async(
    code: str,
    description: str,
    feedback_key: str,
    evaluation_model: Union[BaseChatModel, str, None] = None,
    evaluation_client: Optional[Any] = None,
) -> Tuple[bool, str]:
    """
    Asynchronously evaluates rendered React code with an LLM by taking a screenshot and comparing it to a description.
    
    Args:
        code: The React code to evaluate.
        description: A description of what the React code should look like.
        evaluation_model: The LLM model to use for evaluation.
        evaluation_client: An optional client to use with the evaluation model.
        
    Returns:
        A tuple of (success, comment) where success is a boolean and comment is explanatory text.
    """
    try:
        # Import here to avoid requiring playwright as a hard dependency
        from playwright.async_api import async_playwright
        import base64
        
        # Prepare the React environment
        html_content = _prepare_react_environment(code)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(html_content.encode("utf-8"))
        
        screenshot_path = f"{temp_path}.png"
        
        try:
            # Start playwright and open a browser
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                
                # Navigate to the file with increased timeout
                await page.goto(f"file://{temp_path}", timeout=30000, wait_until="domcontentloaded")
                
                # Wait for page to be loaded
                try:
                    await page.wait_for_selector('body[data-loaded="true"]', timeout=5000)
                except Exception as e:
                    print(f"Warning: Timed out waiting for page load indicator: {e}")
                
                # Ensure wait regardless of selector success
                await page.wait_for_timeout(3000)
                
                # Take a screenshot
                await page.screenshot(path=screenshot_path)
                
                # Close the browser
                await browser.close()
            
            # Read the screenshot as base64
            with open(screenshot_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                
            
            def generate_prompt(description, outputs):
                return [
                    SystemMessage(content="You are an expert React developer evaluating rendered components."),
                    HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": REACT_EVAL_PROMPT.format(description=description),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{outputs}",
                                    "detail": "high",
                                },
                            },
                        ]
                    ),
                ]
            
            judge = create_async_llm_as_judge(
                prompt=generate_prompt,
                feedback_key=feedback_key,
                judge=evaluation_client,
                model=evaluation_model,
            )

            response = await judge(
                outputs=base64_image,
                description=description
            )
            return (response['score'], response['comment'])
            
        finally:
            # Clean up temporary files
            try:
                os.unlink(temp_path)
                if os.path.exists(screenshot_path):
                    os.unlink(screenshot_path)
            except Exception as e:
                print(f"Warning: Error cleaning up temporary files: {e}")
                
    except Exception as e:
        error_details = traceback.format_exc()
        return False, f"Error evaluating React code: {str(e)}\n{error_details}"


def _evaluate_react_code_with_llm(
    code: str,
    inputs: str,
    feedback_key: str,
    evaluation_model: Union[BaseChatModel, str, None] = None,
    evaluation_client: Optional[Any] = None,
) -> Tuple[bool, str]:
    """
    Synchronously evaluates rendered React code with an LLM by taking a screenshot and comparing it to a description.
    
    Args:
        code: The React code to evaluate.
        description: A description of what the React code should look like.
        evaluation_model: The LLM model to use for evaluation.
        evaluation_client: An optional client to use with the evaluation model.
        
    Returns:
        A tuple of (success, comment) where success is a boolean and comment is explanatory text.
    """
    try:
        # Import here to avoid requiring playwright as a hard dependency
        from playwright.sync_api import sync_playwright
        import base64
        
        # Prepare the React environment
        html_content = _prepare_react_environment(code)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(html_content.encode("utf-8"))
        
        screenshot_path = f"{temp_path}.png"
        
        try:    
            # Start playwright and open a browser
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page()
                
                # Navigate to the file with increased timeout
                page.goto(f"file://{temp_path}", timeout=30000, wait_until="domcontentloaded")
                
                # Wait for page to be loaded
                try:
                    page.wait_for_selector('body[data-loaded="true"]', timeout=5000)
                except Exception as e:
                    print(f"Warning: Timed out waiting for page load indicator: {e}")
                
                # Ensure wait regardless of selector success
                page.wait_for_timeout(3000)
                # Take a screenshot
                page.screenshot(path=screenshot_path)
                
                # Close the browser
                browser.close()
            
            # Read the screenshot as base64
            with open(screenshot_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")

            # Create messages for the LLM
            def generate_prompt(inputs, outputs, **kwargs):
                return [
                    SystemMessage(content="You are an expert React developer evaluating rendered components."),
                    HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": REACT_EVAL_PROMPT.format(description=inputs),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{outputs}",
                                    "detail": "high",
                                },
                            },
                        ]
                    ),
                ]
            
            judge = create_llm_as_judge(
                prompt=generate_prompt,
                feedback_key=feedback_key,
                judge=evaluation_client,
                model=evaluation_model,
            )

            response = judge(
                outputs=base64_image,
                inputs=inputs,
            )
            return (response['score'], response['comment'])
            
        finally:
            # Clean up temporary files
            try:
                os.unlink(temp_path)
                if os.path.exists(screenshot_path):
                    os.unlink(screenshot_path)
            except Exception as e:
                print(f"Warning: Error cleaning up temporary files: {e}")
                
    except Exception as e:
        error_details = traceback.format_exc()
        return False, f"Error evaluating React code: {str(e)}\n{error_details}"


def create_react_evaluator(
    code_extraction_strategy: Literal["none", "markdown_code_blocks", "llm"] = "none",
    code_extractor: Optional[Callable[[str], str]] = None,
    extraction_model: Union[BaseChatModel, str, None] = None,
    extraction_client: Optional[Any] = None,
    evaluation_model: Union[BaseChatModel, str, None] = None,
    evaluation_client: Optional[Any] = None,
    feedback_key: str = "react_succeeded",
) -> Callable[[Dict[str, Any]], EvaluatorResult]:
    """
    Creates an evaluator that renders React code and evaluates it using screenshots.
    
    Args:
        extraction_strategy: The strategy to use for extracting code from inputs/outputs.
            'none': Use the raw text as is
            'markdown_code_blocks': Extract code from Markdown code blocks
            'llm': Use an LLM to extract code from text
        code_extractor: An optional function to extract code from text. Takes precedence over extraction_strategy.
        extraction_model: The model to use for code extraction if strategy is 'llm'.
        extraction_client: An optional client to use with the extraction model.
        evaluation_model: The model to use for evaluating screenshots.
        evaluation_client: An optional client to use with the evaluation model.
        feedback_key: The key to use for the feedback in the evaluation result.
        description_field: The field in the input that contains the description of how the React component should look.
    
    Returns:
        An evaluator function that evaluates React code.
    """
    if code_extraction_strategy != "llm" and (extraction_client or extraction_model):
        raise ValueError(
            "extraction_client and extraction_model may only be passed if code_extraction_strategy is 'llm'"
        )

    def _create_scorer(
        evaluation_model: Union[BaseChatModel, str, None] = None,
        evaluation_client: Optional[Any] = None,
    ) -> Callable[[str], Tuple[bool, str]]:
        """
        Creates a scorer function that renders React code and evaluates it.
        
        Args:
            description: A description of what the React code should look like.
            evaluation_model: The model to use for evaluating screenshots.
            evaluation_client: An optional client to use with the evaluation model.
            
        Returns:
            A function that takes code and returns a tuple of (success, comment).
        """
        def scorer(outputs: str, **kwargs) -> Tuple[bool, str]:
            return _evaluate_react_code_with_llm(
                code=outputs,
                inputs=kwargs.get("inputs", ""),
                feedback_key=feedback_key,
                evaluation_model=evaluation_model,
                evaluation_client=evaluation_client,
            )
        return scorer
    
    # Create the scorer with the description
    scorer = _create_scorer(
        evaluation_model=evaluation_model,
        evaluation_client=evaluation_client,
    )
    
    # Use the base code evaluator
    return _create_base_code_evaluator(
        scorer=scorer,
        code_extraction_strategy=code_extraction_strategy,
        code_extractor=code_extractor,
        model=extraction_model,
        client=extraction_client,
        run_name="react_evaluator",
        feedback_key=feedback_key,
    )


async def create_async_react_evaluator(
    code_extraction_strategy: Literal["none", "markdown_code_blocks", "llm"] = "none",
    code_extractor: Optional[Callable[[str], str]] = None,
    extraction_model: Union[BaseChatModel, str, None] = None,
    extraction_client: Optional[Any] = None,
    evaluation_model: Union[BaseChatModel, str, None] = None,
    evaluation_client: Optional[Any] = None,
    feedback_key: str = "react_succeeded",
) -> Callable[[Dict[str, Any]], EvaluatorResult]:
    """
    Creates an async evaluator that renders React code and evaluates it using screenshots.
    
    Args:
        extraction_strategy: The strategy to use for extracting code from inputs/outputs.
            'none': Use the raw text as is
            'markdown_code_blocks': Extract code from Markdown code blocks
            'llm': Use an LLM to extract code from text
        code_extractor: An optional function to extract code from text. Takes precedence over extraction_strategy.
        extraction_model: The model to use for code extraction if strategy is 'llm'.
        extraction_client: An optional client to use with the extraction model.
        evaluation_model: The model to use for evaluating screenshots.
        evaluation_client: An optional client to use with the evaluation model.
        feedback_key: The key to use for the feedback in the evaluation result.
        description_field: The field in the input that contains the description of how the React component should look.
    
    Returns:
        An async evaluator function that evaluates React code.
    """
    if code_extraction_strategy != "llm" and (extraction_client or extraction_model):
        raise ValueError(
            "extraction_client and extraction_model may only be passed if code_extraction_strategy is 'llm'"
        )

    async def _create_async_scorer(
        evaluation_model: Union[BaseChatModel, str, None] = None,
        evaluation_client: Optional[Any] = None,
    ) -> Callable[[str], Tuple[bool, str]]:
        """
        Creates an async scorer function that renders React code and evaluates it.
        
        Args:
            description: A description of what the React code should look like.
            evaluation_model: The model to use for evaluating screenshots.
            evaluation_client: An optional client to use with the evaluation model.
            
        Returns:
            A function that takes code and returns a tuple of (success, comment).
        """
        async def scorer(outputs: str, **kwargs) -> Tuple[bool, str]:
            return await _evaluate_react_code_with_llm_async(
                code=outputs,
                description=kwargs.get("inputs", ""),
                feedback_key=feedback_key,
                evaluation_model=evaluation_model,
                evaluation_client=evaluation_client,
            )
        return scorer
    

    # Create the scorer with the description
    scorer = await _create_async_scorer(
        evaluation_model=evaluation_model,
        evaluation_client=evaluation_client,
    )
    
    # Use the base code evaluator
    return _create_async_base_code_evaluator(
        scorer=scorer,
        code_extraction_strategy=code_extraction_strategy,
        code_extractor=code_extractor,
        model=extraction_model,
        client=extraction_client,
        run_name="react_eval_async",
        feedback_key=feedback_key,
    )