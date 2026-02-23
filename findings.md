Hi friends! Going to drop some interesting findings so please let me know what you think + plus what we should do.

## Setup

We ran LLaMA 3.1 8B through a sycophancy + demographic deference pipeline on 500 legal (MMLU Professional Law) and 500 medical (MedQA USMLE) questions. The basic idea: take questions the model gets right, generate a plausible-but-wrong chain-of-thought for a wrong answer, then have different "personas" assert that wrong answer back to the model and see if it caves. We tested a plain user, a neutral authority (doctor/lawyer), authority with demographics (race x gender, gender-only, race-only), and also injecting demographics into the patient/defendant.

## Finding 1: The model is extremely sycophantic

On questions it originally answered correctly, it flipped to the wrong answer **55% of the time in legal** and **74% in medical** when a plain user pushed back. It often literally writes "I disagree with this reasoning" and then picks the wrong answer anyway.

## Finding 2: Adding an authority title actually reduces sycophancy

"A lawyer" gets 32% deference vs 55% for "a user" in legal. Same pattern in medical (58% vs 74%). The model seems to engage more critically when it knows it's talking to an expert.

## Finding 3: Demographic attributes on the authority figure change deference rates, and the pattern differs by domain

### Legal domain

- **Gender-only is significant (p=0.037):** "A female lawyer" gets 34.1% deference vs 25.5% for "a male lawyer" -- an 8.6pp gap
- **Race-only is NOT significant (p=1.00):** "A White lawyer" (26.6%) and "A Black lawyer" (26.2%) are basically identical
- **Race x gender combined IS significant (p=0.008):** White male lawyer gets the least deference (37.8%), everyone else clusters around 46-51% (White female 46.4%, Black male 50.9%, Black female 50.6%)
- So gender alone drives a significant effect, race alone does not, but when both are specified White male is the clear outlier

### Medical domain

- **Race-only is highly significant (p<0.0001):** "A Black doctor" gets 70.3% deference vs 50.7% for "A White doctor" -- a massive 20pp gap. This is the strongest finding in the entire experiment (Cramer's V=0.197)
- **Gender-only trends but is not significant (p=0.082):** female doctor 64.4% vs male doctor 57.2%
- **Race x gender combined is NOT significant (p=0.107):** the effect gets diluted when both attributes are specified
- Opposite pattern from legal: here race is what matters, not gender

## Finding 4: Person demographics (patient/defendant) don't matter

Changing the race/gender of the person in the question has essentially no effect on whether the model caves. Not significant in either domain (legal p=0.99, medical p=0.72). This is actually somewhat reassuring -- the model isn't giving different clinical/legal reasoning based on who the patient or defendant is.

## Key takeaway

The bias isn't "the model distrusts minority experts." It's the opposite -- the model is **less willing to push back** against non-default demographic groups. It treats "White male lawyer" almost identically to just "a lawyer," and any other demographic marker makes it more deferential. This looks like safety training overcorrection -- the model has learned to be extra agreeable when interacting with people from underrepresented groups, which ironically means it gives them worse advice.

## Open questions

1. Should we run a second model (e.g. GPT-4o or Mistral) to see if this is LLaMA-specific or a general pattern?
2. Do we want to expand the demographic matrix (add Asian, Hispanic, etc.)?
3. How do we want to frame the "overcorrection" angle in the writeup -- this is a sensitive finding
4. Should we try a debiasing prompt to see if we can reduce the sycophancy?

Lmk what you think!
