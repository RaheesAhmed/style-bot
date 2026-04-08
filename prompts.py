"""System prompts and prompt templates for the Style-Aware AI Text Generator.

This module contains all prompt engineering logic including system prompts,
user prompts, and ChatPromptTemplate definitions.
"""

from langchain_core.prompts import ChatPromptTemplate


# System prompt for style mimicry
STYLE_SYSTEM_PROMPT = """You are an expert writing assistant. Your ONLY job is to generate content that strictly matches the user's unique writing style, tone, structure, and phrasing.

Analyze the provided examples carefully. Mimic their:
- Sentence length & rhythm
- Vocabulary, idioms, and technical depth
- Tone (formal, casual, authoritative, conversational, etc.)
- Formatting, bullet usage, and paragraph structure

NEVER output generic AI text. If web research is provided, weave it naturally into the established style.
Output ONLY the generated content. No intros, no conclusions, no explanations."""

# User prompt template
STYLE_USER_PROMPT = """TASK: {task}

STYLE EXAMPLES FROM ARCHIVE:
{context}

{web_search_context}

GENERATE:"""


def create_style_prompt() -> ChatPromptTemplate:
    """Create the ChatPromptTemplate for style-aware generation.
    
    Returns:
        ChatPromptTemplate: Configured prompt template for the agent.
    """
    return ChatPromptTemplate.from_messages([
        ("system", STYLE_SYSTEM_PROMPT),
        ("human", STYLE_USER_PROMPT),
    ])


def format_context_message(context: str, web_context: str = "") -> str:
    """Format the context message with optional web research.
    
    Args:
        context: Style examples from the vector database.
        web_context: Optional web research context.
        
    Returns:
        str: Formatted context message.
    """
    if web_context:
        return f"{context}\n\n{web_context}"
    return context