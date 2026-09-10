"""Shared config and client helpers used across the src/ scripts."""

import os

from dotenv import load_dotenv
from langsmith import Client
from langsmith.wrappers import wrap_openai
from openai import OpenAI

load_dotenv()

DATASET_NAME = "bank-transaction-categorization-v1"
DATASET_DESCRIPTION = (
    "Raw bank-statement transaction lines (synthetic, EU-style exports) paired "
    "with the analyst-assigned category and merchant. Used to evaluate an LLM's "
    "ability to categorize noisy transaction text the way a real Enable Banking / "
    "PSD2 feed would return it."
)
MODEL_NAME = "gpt-4o-mini"
JUDGE_MODEL_NAME = "gpt-4o-mini"
EXPERIMENT_PREFIX = "txn-categorization"

CATEGORIES = [
    "Groceries",
    "Dining & Restaurants",
    "Transport",
    "Utilities",
    "Rent & Housing",
    "Subscriptions",
    "Shopping",
    "Healthcare",
    "Entertainment",
    "Travel",
    "Income & Salary",
    "Transfers",
    "Fees & Charges",
    "Insurance",
    "Other",
]


def get_langsmith_client() -> Client:
    return Client(
        api_key=os.environ["LANGSMITH_API_KEY"],
        api_url=os.environ.get("LANGSMITH_ENDPOINT"),
    )


def get_openai_client() -> OpenAI:
    """Raw OpenAI client wrapped so every call is auto-traced to LangSmith."""
    return wrap_openai(OpenAI(api_key=os.environ["OPENAI_API_KEY"]))
