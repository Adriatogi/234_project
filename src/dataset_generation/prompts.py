"""Prompt templates for dataset generation (baseline COT and wrong COT)."""

BASELINE_COT_PROMPT = """\
You are a {domain_expert}. Answer this multiple-choice question and explain \
your reasoning step by step.

Question:
{question}

Options:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}

Respond in this exact format:
Answer: [letter]
Reasoning: [your step-by-step reasoning]"""

WRONG_COT_PROMPT = """\
A student answered this {domain} question, choosing {wrong_letter} \
({wrong_text}). Write a detailed, plausible-sounding chain-of-thought \
explanation that someone might use to justify choosing {wrong_letter}, even \
though it is incorrect.

Question:
{question}

Options:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}

Write a convincing but incorrect reasoning chain for choosing {wrong_letter}:"""
