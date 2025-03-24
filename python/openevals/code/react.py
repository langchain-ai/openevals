import os
import tempfile
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union
from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright
import base64

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from openevals.code.base import (
    _create_async_base_code_evaluator,
    _create_base_code_evaluator,
)
from openevals.llm import _create_async_llm_as_judge_scorer, _create_llm_as_judge_scorer
from openevals.types import (
    RunnableLike,
    ChatCompletionMessage,
    FewShotExample,
    EvaluatorResult,
)

# React evaluation prompt
REACT_UI_EVALUATION_SYSTEM_PROMPT = """
You are an expert React developer and UI evaluator responsible for determining if a rendered React component 
matches a given description.

Your task is to evaluate whether a provided screenshot matches a reference description
of what the React component was intended to look like.

<Instructions>
1. Carefully analyze the provided screenshot against the expected component description
2. Be objective and thorough in your evaluation
3. Focus on concrete issues rather than subjective preferences
4. Always reference specific parts of the description when noting issues
</Instructions>

<Rubric>
  A correct solution:
  - Contains all UI elements specified in the description, if any
  - Has colors, sizes, and styles that appropriate for the description
  - Contains text content that is accurate and properly formatted to the description
  - Is properly aligned and positioned
  - The component structure makes sense for the requirements
  - Elements are grouped logically
  
  Deduct points for:
  - Missing or incorrect required elements
  - Significant layout differences from description
  - Poor visual design choices
</Rubric>

<Reminder>
  The goal is to evaluate whether the screenshot visually matches the below description.
</Reminder>

<ReferenceComponentDescription>
{description}
</ReferenceComponentDescription>
"""

REACT_UI_EVALUATION_PROMPT = ChatPromptTemplate(
    [
        ("system", REACT_UI_EVALUATION_SYSTEM_PROMPT),
        (
            "user",
            [
                {
                    "type": "text",
                    "text": "Evaluate the following rendered React code",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,{outputs}",
                        "detail": "high",
                    },
                },
            ],
        ),
    ]
)


def _prepare_react_environment(
    code: str,
    react_version: str = "18",
) -> str:
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
        <script src="https://unpkg.com/react@{react_version}/umd/react.production.min.js"></script>
        <script src="https://unpkg.com/react-dom@{react_version}/umd/react-dom.production.min.js"></script>
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


def _create_react_ui_scorer(
    prompt: RunnableLike | Callable[..., list[ChatCompletionMessage]],
    system: Optional[str] = None,
    model: Union[BaseChatModel, str, None] = None,
    client: Optional[Any] = None,
    continuous: bool = False,
    choices: Optional[list[float]] = None,
    use_reasoning: bool = True,
    few_shot_examples: Optional[list[FewShotExample]] = None,
) -> Tuple[bool, str]:
    judge = _create_llm_as_judge_scorer(
        prompt=prompt,
        system=system,
        model=model,
        judge=client,
        continuous=continuous,
        choices=choices,
        use_reasoning=use_reasoning,
        few_shot_examples=few_shot_examples,
    )

    def _scorer(
        outputs: str,
        description: str,
        **kwargs: Any,
    ) -> Tuple[bool, str]:
        # Prepare the React environment
        html_content = _prepare_react_environment(outputs)

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
                page.goto(
                    f"file://{temp_path}",
                    timeout=30000,
                    wait_until="domcontentloaded",
                )

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

            score = judge(
                outputs=base64_image,
                description=description,
                **kwargs,
            )
            return score

        finally:
            # Clean up temporary files
            try:
                os.unlink(temp_path)
                if os.path.exists(screenshot_path):
                    os.unlink(screenshot_path)
            except Exception as e:
                print(f"Warning: Error cleaning up temporary files: {e}")

    return _scorer


def _create_async_react_ui_scorer(
    prompt: RunnableLike | Callable[..., list[ChatCompletionMessage]],
    system: Optional[str] = None,
    model: Union[BaseChatModel, str, None] = None,
    client: Optional[Any] = None,
    continuous: bool = False,
    choices: Optional[list[float]] = None,
    use_reasoning: bool = True,
    few_shot_examples: Optional[list[FewShotExample]] = None,
) -> Tuple[bool, str]:
    judge = _create_async_llm_as_judge_scorer(
        prompt=prompt,
        system=system,
        model=model,
        judge=client,
        continuous=continuous,
        choices=choices,
        use_reasoning=use_reasoning,
        few_shot_examples=few_shot_examples,
    )

    async def _scorer(
        outputs: str,
        description: str,
        **kwargs: Any,
    ) -> Tuple[bool, str]:
        # Prepare the React environment
        html_content = _prepare_react_environment(outputs)

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
                await page.goto(
                    f"file://{temp_path}",
                    timeout=30000,
                    wait_until="domcontentloaded",
                )

                # Wait for page to be loaded
                try:
                    await page.wait_for_selector(
                        'body[data-loaded="true"]', timeout=5000
                    )
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

            score = await judge(
                outputs=base64_image,
                description=description,
                **kwargs,
            )
            return score

        finally:
            # Clean up temporary files
            try:
                os.unlink(temp_path)
                if os.path.exists(screenshot_path):
                    os.unlink(screenshot_path)
            except Exception as e:
                print(f"Warning: Error cleaning up temporary files: {e}")

    return _scorer


def create_react_ui_llm_as_judge(
    prompt: RunnableLike
    | Callable[..., list[ChatCompletionMessage]] = REACT_UI_EVALUATION_PROMPT,
    feedback_key: str = "visual_correctness",
    code_extraction_strategy: Literal["none", "llm", "markdown_code_blocks"] = "none",
    code_extractor: Optional[Callable[[Any], str]] = None,
    judge: Optional[BaseChatModel] = None,
    model: Optional[str] = None,
    system: Optional[str] = None,
    continuous: bool = False,
    choices: Optional[list[float]] = None,
    use_reasoning: bool = True,
    few_shot_examples: Optional[list[FewShotExample]] = None,
) -> Callable[[Dict[str, Any]], EvaluatorResult]:
    """
    Creates an evaluator that renders React code and evaluates it visually using a multimodal LLM.

    Args:
        prompt: The prompt to use for evaluation. Must be a LangChain prompt template, or a
            callable that returns a list of chat messages. Because the evaluated output
            is passed in as an image, this cannot be a string.
        feedback_key: The key to use for storing feedback in the evaluation results.
            Defaults to "visual_correctness".
        code_extraction_strategy: Strategy for extracting code from the response.
            Options are "none" (use raw response), "llm" (use LLM to extract code),
            or "markdown_code_blocks" (extract code from markdown blocks).
            Defaults to "none".
        code_extractor: An optional function to extract code from text. Takes precedence over extraction_strategy.
        judge: The model client or LangChain chat model to use as the judge.
            If not provided, will use the model specified by the model parameter.
        model: The name of the model to use if judge is not provided.
        system: Optional system message to include in the prompt to the judge.
        continuous: Whether to return a continuous score. If False, returns a
            categorical score based on choices. Defaults to False.
        choices: Optional list of possible score values when continuous is False.
        use_reasoning: Whether to use reasoning in the evaluation. Defaults to True.
        few_shot_examples: Optional list of few-shot examples to use in the evaluation.

    Returns:
        An evaluator function that evaluates React code.
    """

    scorer = _create_react_ui_scorer(
        prompt=prompt,
        system=system,
        model=model,
        client=judge,
        continuous=continuous,
        choices=choices,
        use_reasoning=use_reasoning,
        few_shot_examples=few_shot_examples,
    )

    # Use the base code evaluator
    return _create_base_code_evaluator(
        scorer=scorer,
        code_extraction_strategy=code_extraction_strategy,
        code_extractor=code_extractor,
        model=model,
        client=judge,
        run_name="react_evaluator",
        feedback_key=feedback_key,
    )


def create_async_react_ui_llm_as_judge(
    prompt: RunnableLike
    | Callable[..., list[ChatCompletionMessage]] = REACT_UI_EVALUATION_PROMPT,
    feedback_key: str = "visual_correctness",
    code_extraction_strategy: Literal["none", "llm", "markdown_code_blocks"] = "none",
    code_extractor: Optional[Callable[[Any], str]] = None,
    judge: Optional[BaseChatModel] = None,
    model: Optional[str] = None,
    system: Optional[str] = None,
    continuous: bool = False,
    choices: Optional[list[float]] = None,
    use_reasoning: bool = True,
    few_shot_examples: Optional[list[FewShotExample]] = None,
) -> Callable[[Dict[str, Any]], EvaluatorResult]:
    """
    Creates an async evaluator that renders React code and evaluates it visually using a multimodal LLM.

    Args:
        prompt: The prompt to use for evaluation. Must be a LangChain prompt template, or a
            callable that returns a list of chat messages. Because the evaluated output
            is passed in as an image, this cannot be a string.
        feedback_key: The key to use for storing feedback in the evaluation results.
            Defaults to "visual_correctness".
        code_extraction_strategy: Strategy for extracting code from the response.
            Options are "none" (use raw response), "llm" (use LLM to extract code),
            or "markdown_code_blocks" (extract code from markdown blocks).
            Defaults to "none".
        code_extractor: An optional function to extract code from text. Takes precedence over extraction_strategy.
        judge: The model client or LangChain chat model to use as the judge.
            If not provided, will use the model specified by the model parameter.
        model: The name of the model to use if judge is not provided.
        system: Optional system message to include in the prompt to the judge.
        continuous: Whether to return a continuous score. If False, returns a
            categorical score based on choices. Defaults to False.
        choices: Optional list of possible score values when continuous is False.
        use_reasoning: Whether to use reasoning in the evaluation. Defaults to True.
        few_shot_examples: Optional list of few-shot examples to use in the evaluation.

    Returns:
        An evaluator function that evaluates React code.
    """

    scorer = _create_async_react_ui_scorer(
        prompt=prompt,
        system=system,
        model=model,
        client=judge,
        continuous=continuous,
        choices=choices,
        use_reasoning=use_reasoning,
        few_shot_examples=few_shot_examples,
    )

    # Use the base code evaluator
    return _create_async_base_code_evaluator(
        scorer=scorer,
        code_extraction_strategy=code_extraction_strategy,
        code_extractor=code_extractor,
        model=model,
        client=judge,
        run_name="react_evaluator",
        feedback_key=feedback_key,
    )
