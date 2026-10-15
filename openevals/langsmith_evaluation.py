"""
LangSmith Custom Evaluation Pipeline
======================================
Domain: UX Design Knowledge Q&A

This script implements the full evaluation workflow:
  1. Creates a custom LangSmith dataset with 15 UX design Q&A examples
  2. Implements a target function (with @traceable) that answers UX questions
  3. Sets up a correctness evaluator using LLM-as-judge
  4. Runs the evaluation experiment via LangSmith
  5. Analyzes results and saves a summary

Usage:
    pip install openai langsmith langchain-openai openevals python-dotenv
    Create .env with OPENAI_API_KEY and LANGSMITH_API_KEY
    python langsmith_evaluation.py
"""

import json
import os
import sys
import time
from datetime import datetime

from dotenv import load_dotenv
from langsmith import Client, evaluate, traceable
from openai import OpenAI

# ---------------------------------------------------------------------------
# 1. Setup & Environment
# ---------------------------------------------------------------------------
load_dotenv()

# LangSmith configuration
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "ux-design-eval")

openai_key = os.environ.get("OPENAI_API_KEY")
langsmith_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://eu.api.smith.langchain.com")
if not openai_key:
    raise RuntimeError("OPENAI_API_KEY not found in environment or .env")
if not langsmith_key:
    raise RuntimeError(
        "LANGSMITH_API_KEY not found in environment or .env. "
        "Create a LangSmith API key and set it in your .env file."
    )

os.environ["LANGSMITH_API_KEY"] = langsmith_key
os.environ["LANGCHAIN_API_KEY"] = langsmith_key

ls_client = Client(api_key=langsmith_key)
oai_client = OpenAI(api_key=openai_key)

MODEL = "gpt-4o-mini"
DATASET_NAME = "ux-design-knowledge-qa"
EXPERIMENT_PREFIX = "ux-eval-gpt4o-mini"


def ensure_langsmith_access():
    """Verify that the configured LangSmith key can read datasets for this workspace."""
    try:
        list(ls_client.list_datasets(limit=1))
    except Exception as exc:  # pragma: no cover - network/auth failure path
        masked_key = (langsmith_key[:6] + "..." + langsmith_key[-4:]) if len(langsmith_key) > 10 else "***"
        raise RuntimeError(
            "LangSmith access check failed. Your API key is invalid, expired, or does not have access "
            "to the active workspace.\n"
            f"  Current key: {masked_key}\n"
            "  Fix: create a new LangSmith API key, ensure it belongs to the same account/workspace, "
            "and verify the key is set as LANGSMITH_API_KEY (or LANGCHAIN_API_KEY)."
        ) from exc


print("Setup complete.")
print(f"  LangSmith project: {os.environ.get('LANGCHAIN_PROJECT')}")
print(f"  Model: {MODEL}")
print(f"  Dataset: {DATASET_NAME}")

# ---------------------------------------------------------------------------
# 2. Dataset — 15 UX Design Knowledge Q&A examples
# ---------------------------------------------------------------------------
EXAMPLES = [
    {
        "inputs": {
            "question": "What are Nielsen's 10 usability heuristics? List them briefly."
        },
        "outputs": {
            "answer": (
                "Nielsen's 10 usability heuristics are: (1) Visibility of system status, "
                "(2) Match between system and the real world, (3) User control and freedom, "
                "(4) Consistency and standards, (5) Error prevention, (6) Recognition rather "
                "than recall, (7) Flexibility and efficiency of use, (8) Aesthetic and minimalist "
                "design, (9) Help users recognize, diagnose, and recover from errors, "
                "(10) Help and documentation."
            )
        },
        "metadata": {"category": "heuristics", "difficulty": "medium"},
    },
    {
        "inputs": {
            "question": "What is the difference between UX and UI design?"
        },
        "outputs": {
            "answer": (
                "UX (User Experience) design focuses on the overall feel and usability of a product — "
                "how users interact with it, how easy it is to accomplish tasks, and how satisfying "
                "the experience is. UI (User Interface) design focuses on the visual and interactive "
                "elements — layout, typography, color, buttons, and icons. UX is about the journey; "
                "UI is about the look and feel of each touchpoint along that journey."
            )
        },
        "metadata": {"category": "fundamentals", "difficulty": "easy"},
    },
    {
        "inputs": {
            "question": "What is a design system and why is it important?"
        },
        "outputs": {
            "answer": (
                "A design system is a collection of reusable components, patterns, and guidelines "
                "that ensure consistency across a product or suite of products. It typically includes "
                "a component library, design tokens (colors, typography, spacing), interaction patterns, "
                "and documentation. It is important because it speeds up design and development, "
                "ensures visual and behavioral consistency, reduces redundant work, and makes "
                "onboarding easier for new team members."
            )
        },
        "metadata": {"category": "design_systems", "difficulty": "medium"},
    },
    {
        "inputs": {
            "question": "What does WCAG stand for and what are its conformance levels?"
        },
        "outputs": {
            "answer": (
                "WCAG stands for Web Content Accessibility Guidelines. It has three conformance "
                "levels: Level A (minimum accessibility), Level AA (addresses the most common "
                "barriers and is the standard most organizations target), and Level AAA (the highest "
                "level of accessibility, which is not required for entire sites as it may not be "
                "achievable for all content)."
            )
        },
        "metadata": {"category": "accessibility", "difficulty": "medium"},
    },
    {
        "inputs": {
            "question": "What is the minimum contrast ratio for normal text under WCAG 2.1 AA?"
        },
        "outputs": {
            "answer": (
                "Under WCAG 2.1 Level AA, the minimum contrast ratio for normal-sized text "
                "(below 18pt or 14pt bold) is 4.5:1 against its background. For large text "
                "(18pt and above, or 14pt bold and above), the minimum ratio is 3:1."
            )
        },
        "metadata": {"category": "accessibility", "difficulty": "hard"},
    },
    {
        "inputs": {
            "question": "Explain Fitts's Law and how it applies to UI design."
        },
        "outputs": {
            "answer": (
                "Fitts's Law states that the time to reach a target is a function of the distance "
                "to the target and the size of the target. In UI design, this means: make frequently "
                "used buttons and interactive elements larger and place them closer to where the "
                "user's cursor or finger is likely to be. For example, primary action buttons should "
                "be large and prominent, and navigation elements should be placed at screen edges "
                "where they are easiest to reach."
            )
        },
        "metadata": {"category": "interaction_design", "difficulty": "medium"},
    },
    {
        "inputs": {
            "question": "What is a user persona and what should it include?"
        },
        "outputs": {
            "answer": (
                "A user persona is a fictional but research-based representation of a key user "
                "segment. It typically includes: a name and photo, demographic information, goals "
                "and motivations, pain points and frustrations, behaviors and habits, a brief "
                "scenario or quote, and context about how they would use the product. Personas "
                "help teams make user-centered design decisions by keeping a concrete user in mind."
            )
        },
        "metadata": {"category": "user_research", "difficulty": "easy"},
    },
    {
        "inputs": {
            "question": "What is the difference between formative and summative usability testing?"
        },
        "outputs": {
            "answer": (
                "Formative usability testing is conducted during the design process to identify "
                "problems and improve the design iteratively. It is typically qualitative, uses "
                "think-aloud protocols, and requires fewer participants (5-8). Summative usability "
                "testing is conducted after a design is complete to measure overall usability with "
                "metrics like task success rate, time on task, and error rate. It is more quantitative "
                "and requires larger sample sizes for statistical validity."
            )
        },
        "metadata": {"category": "user_research", "difficulty": "hard"},
    },
    {
        "inputs": {
            "question": "What is information architecture (IA) in UX design?"
        },
        "outputs": {
            "answer": (
                "Information architecture is the practice of organizing, structuring, and labeling "
                "content in a way that helps users find information and complete tasks. It involves "
                "creating site maps, navigation structures, taxonomies, and labeling systems. Good "
                "IA makes a product intuitive by grouping related content logically and providing "
                "clear navigation paths. Common IA methods include card sorting, tree testing, "
                "and content audits."
            )
        },
        "metadata": {"category": "information_architecture", "difficulty": "medium"},
    },
    {
        "inputs": {
            "question": "What is the difference between low-fidelity and high-fidelity prototypes?"
        },
        "outputs": {
            "answer": (
                "Low-fidelity prototypes are quick, rough representations of a design — such as "
                "paper sketches or simple wireframes — used early in the design process to explore "
                "ideas and get fast feedback. They focus on layout and flow, not visual polish. "
                "High-fidelity prototypes closely resemble the final product with realistic visuals, "
                "interactions, and content. They are used later in the process for detailed usability "
                "testing and stakeholder sign-off. Low-fi is fast and cheap; high-fi is slower but "
                "more realistic."
            )
        },
        "metadata": {"category": "prototyping", "difficulty": "easy"},
    },
    {
        "inputs": {
            "question": "What is a heuristic evaluation and how is it conducted?"
        },
        "outputs": {
            "answer": (
                "A heuristic evaluation is an inspection method where evaluators (typically 3-5 UX "
                "experts) examine a user interface against a set of recognized usability principles "
                "(heuristics), such as Nielsen's 10 heuristics. Each evaluator independently reviews "
                "the interface, identifies violations, and rates their severity. Findings are then "
                "consolidated into a report with prioritized recommendations. It is a cost-effective "
                "way to find usability issues without recruiting test participants."
            )
        },
        "metadata": {"category": "evaluation_methods", "difficulty": "medium"},
    },
    {
        "inputs": {
            "question": "What is the purpose of a design token?"
        },
        "outputs": {
            "answer": (
                "Design tokens are named entities that store visual design attributes — such as "
                "colors, font sizes, spacing, border radii, and shadows — as platform-agnostic "
                "values. Their purpose is to create a single source of truth for design decisions "
                "that can be shared across design tools (like Figma) and code (CSS, iOS, Android). "
                "When a token value changes, the update propagates everywhere it is used, ensuring "
                "consistency and making theming (like dark mode) much easier to implement."
            )
        },
        "metadata": {"category": "design_systems", "difficulty": "medium"},
    },
    {
        "inputs": {
            "question": "What is cognitive load in UX, and how can designers reduce it?"
        },
        "outputs": {
            "answer": (
                "Cognitive load is the amount of mental effort required to use a product. High "
                "cognitive load makes interfaces feel confusing and exhausting. Designers can reduce "
                "it by: simplifying choices (Hick's Law — fewer options mean faster decisions), "
                "chunking information into smaller groups, using progressive disclosure to show only "
                "what is needed at each step, maintaining visual hierarchy to guide attention, using "
                "familiar patterns and conventions, and minimizing the need for users to remember "
                "information across steps."
            )
        },
        "metadata": {"category": "psychology", "difficulty": "medium"},
    },
    {
        "inputs": {
            "question": "What is the System Usability Scale (SUS)?"
        },
        "outputs": {
            "answer": (
                "The System Usability Scale is a standardized questionnaire consisting of 10 "
                "statements rated on a 5-point Likert scale (Strongly Disagree to Strongly Agree). "
                "It produces a single score between 0 and 100 that represents the perceived usability "
                "of a system. A SUS score above 68 is considered above average. It was created by "
                "John Brooke in 1986 and remains one of the most widely used usability metrics "
                "because it is quick, reliable, and technology-agnostic."
            )
        },
        "metadata": {"category": "evaluation_methods", "difficulty": "hard"},
    },
    {
        "inputs": {
            "question": "What are the key principles of Apple's Human Interface Guidelines (HIG) for iOS?"
        },
        "outputs": {
            "answer": (
                "Apple's HIG emphasizes several core principles: clarity (text is legible, icons are "
                "precise, functionality is obvious), deference (the UI helps people understand and "
                "interact with content without competing with it), and depth (visual layers and "
                "realistic motion convey hierarchy and facilitate understanding). Additional guidelines "
                "cover consistency with platform conventions, direct manipulation, feedback, user "
                "control, aesthetic integrity, and accessibility. Designers should use standard iOS "
                "components and patterns to meet user expectations."
            )
        },
        "metadata": {"category": "platform_guidelines", "difficulty": "hard"},
    },
]


def create_dataset():
    """Create (or retrieve) the LangSmith dataset and upload examples."""
    # Check if dataset already exists
    existing = list(ls_client.list_datasets(dataset_name=DATASET_NAME))
    if existing:
        dataset = existing[0]
        print(f"\nDataset '{DATASET_NAME}' already exists (id: {dataset.id}). Skipping creation.")
        return dataset

    print(f"\nCreating dataset '{DATASET_NAME}' with {len(EXAMPLES)} examples...")
    dataset = ls_client.create_dataset(
        dataset_name=DATASET_NAME,
        description=(
            "UX Design Knowledge Q&A — 15 questions covering usability heuristics, "
            "accessibility (WCAG), design systems, user research methods, interaction "
            "design, information architecture, and platform guidelines. Each example "
            "includes a question and a reference answer for correctness evaluation."
        ),
    )

    for i, ex in enumerate(EXAMPLES, 1):
        ls_client.create_example(
            inputs=ex["inputs"],
            outputs=ex["outputs"],
            dataset_id=dataset.id,
            metadata=ex.get("metadata", {}),
        )

    print(f"  Uploaded {len(EXAMPLES)} examples to dataset (id: {dataset.id}).")
    return dataset


# ---------------------------------------------------------------------------
# 3. Target Function — answers UX design questions
# ---------------------------------------------------------------------------
@traceable
def ux_qa_target(inputs: dict) -> dict:
    """
    Target function: takes a UX design question and returns an answer
    using GPT-4o-mini with a domain-specific system prompt.
    """
    question = inputs["question"]

    response = oai_client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior UX design expert. Answer the following question "
                    "accurately and concisely. Focus on practical, actionable knowledge. "
                    "If the question asks for a list, provide all items. Be precise with "
                    "numbers and standards (e.g., WCAG ratios, heuristic names)."
                ),
            },
            {"role": "user", "content": question},
        ],
    )

    answer = response.choices[0].message.content
    return {"answer": answer}


# ---------------------------------------------------------------------------
# 4. Evaluator — LLM-as-judge correctness
# ---------------------------------------------------------------------------
def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """
    LLM-as-judge evaluator that checks whether the model's answer
    is factually correct and complete compared to the reference answer.
    Returns a score (0 or 1) and reasoning.
    """
    question = inputs.get("question", "")
    model_answer = outputs.get("answer", "")
    reference_answer = reference_outputs.get("answer", "")

    eval_prompt = f"""You are an expert UX design instructor evaluating a student's answer.

QUESTION:
{question}

REFERENCE ANSWER (ground truth):
{reference_answer}

STUDENT ANSWER (to evaluate):
{model_answer}

Evaluate whether the student's answer is factually correct and sufficiently complete.
The student does NOT need to match the reference word-for-word, but must:
1. Include all key facts and concepts from the reference
2. Not contain factual errors or hallucinations
3. Be relevant and on-topic

Respond with JSON only (no markdown backticks):
{{"score": 1 or 0, "reasoning": "brief explanation"}}

Score 1 = correct and complete. Score 0 = has factual errors or missing critical information."""

    response = oai_client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a strict but fair evaluator. Respond with JSON only."},
            {"role": "user", "content": eval_prompt},
        ],
    )

    raw = response.choices[0].message.content
    try:
        result = json.loads(raw)
        return {
            "key": "correctness",
            "score": result.get("score", 0),
            "comment": result.get("reasoning", ""),
        }
    except json.JSONDecodeError:
        return {
            "key": "correctness",
            "score": 0,
            "comment": f"Failed to parse judge output: {raw[:200]}",
        }


# ---------------------------------------------------------------------------
# 5. Run Evaluation
# ---------------------------------------------------------------------------
def run_evaluation(dataset):
    """Run the evaluation experiment via LangSmith."""
    print(f"\nRunning evaluation experiment: '{EXPERIMENT_PREFIX}'...")
    print(f"  Dataset: {DATASET_NAME} ({len(EXAMPLES)} examples)")
    print(f"  Model: {MODEL}")
    print(f"  Evaluator: correctness (LLM-as-judge)")
    print("  This may take 1-2 minutes...\n")

    start = time.time()

    results = evaluate(
        ux_qa_target,
        data=DATASET_NAME,
        evaluators=[correctness_evaluator],
        experiment_prefix=EXPERIMENT_PREFIX,
        max_concurrency=2,
    )

    elapsed = round(time.time() - start, 2)
    print(f"\nEvaluation completed in {elapsed}s.")
    return results, elapsed


# ---------------------------------------------------------------------------
# 6. Analyze Results
# ---------------------------------------------------------------------------
def analyze_results(results, elapsed):
    """Analyze evaluation results and save to JSON."""
    scores = []
    details = []

    for result in results:
        score = None
        reasoning = ""
        question = ""

        # Try multiple ways to extract data from the result
        try:
            # Get question
            if isinstance(result, dict):
                run = result.get("run")
                example = result.get("example")
                eval_res = result.get("evaluation_results")
            else:
                run = getattr(result, "run", None)
                example = getattr(result, "example", None)
                eval_res = getattr(result, "evaluation_results", None)

            # Extract question from run or example.
            # The target takes an `inputs` dict, so a traced run nests it one level.
            if run and hasattr(run, "inputs"):
                run_inputs = run.inputs or {}
                question = run_inputs.get("question") or run_inputs.get("inputs", {}).get("question", "")
            if not question and example and hasattr(example, "inputs"):
                question = (example.inputs or {}).get("question", "")

            # Extract score from evaluation results
            if eval_res:
                res_list = None
                if isinstance(eval_res, dict):
                    res_list = eval_res.get("results", [])
                elif hasattr(eval_res, "results"):
                    res_list = eval_res.results
                
                if res_list:
                    for r in res_list:
                        s = r.get("score") if isinstance(r, dict) else getattr(r, "score", None)
                        c = r.get("comment", "") if isinstance(r, dict) else getattr(r, "comment", "")
                        if s is not None:
                            score = s
                        if c:
                            reasoning = c
        except Exception as e:
            print(f"  Warning: Could not parse result: {e}")

        if score is not None:
            scores.append(score)

        question = question[:80] + "..." if len(question) > 80 else question

        details.append({
            "question": question,
            "score": score,
            "reasoning": reasoning[:150] if reasoning else "",
        })

    # Aggregate metrics
    if scores:
        avg_score = round(sum(scores) / len(scores), 2)
        pass_count = sum(1 for s in scores if s == 1)
        fail_count = sum(1 for s in scores if s == 0)
        pass_rate = round(pass_count / len(scores) * 100, 1)
    else:
        avg_score = 0
        pass_count = 0
        fail_count = 0
        pass_rate = 0

    analysis = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": MODEL,
            "dataset": DATASET_NAME,
            "experiment_prefix": EXPERIMENT_PREFIX,
            "total_examples": len(EXAMPLES),
            "total_time_seconds": elapsed,
        },
        "aggregate_metrics": {
            "average_score": avg_score,
            "pass_rate_percent": pass_rate,
            "passed": pass_count,
            "failed": fail_count,
            "total_evaluated": len(scores),
        },
        "per_example_results": details,
    }

    # Save to JSON
    with open("evaluation_results.json", "w") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Model:           {MODEL}")
    print(f"  Dataset:         {DATASET_NAME}")
    print(f"  Examples:        {len(EXAMPLES)}")
    print(f"  Total time:      {elapsed}s")
    print(f"  Average score:   {avg_score}")
    print(f"  Pass rate:       {pass_rate}% ({pass_count}/{len(scores)})")
    print()

    for d in details:
        status = "PASS" if d["score"] == 1 else "FAIL" if d["score"] == 0 else "N/A"
        print(f"  [{status}] {d['question']}")
        if d["reasoning"]:
            print(f"        {d['reasoning']}")

    print("=" * 60)
    print(f"\nResults saved to evaluation_results.json")
    print(f"View experiment in LangSmith UI under project: {os.environ.get('LANGCHAIN_PROJECT')}")

    return analysis


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        ensure_langsmith_access()

        # Step 1: Create dataset
        dataset = create_dataset()

        # Step 2-3: Target function and evaluator are defined above

        # Step 4: Run evaluation
        results, elapsed = run_evaluation(dataset)

        # Step 5: Analyze
        analyze_results(results, elapsed)
    except Exception as exc:
        print("\nEvaluation setup failed.")
        print(f"Reason: {exc}")
        print("\nChecklist:")
        print("  1. Regenerate your LangSmith API key in the LangSmith dashboard.")
        print("  2. Ensure it belongs to the same workspace/account you are using.")
        print("  3. Confirm the API key is in your .env as LANGSMITH_API_KEY.")
        print("  4. Re-run: python langsmith_evaluation.py")
        sys.exit(1)
