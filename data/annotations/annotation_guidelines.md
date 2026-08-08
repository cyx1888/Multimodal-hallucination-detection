# Annotation Guidelines

## Purpose
You are evaluating whether a multimodal model's answer contains hallucination.
Judge based ONLY on: the image, question, ground-truth answer, and model answer.

## Hallucination Types

### 1. Visual Grounding Error
The model mentions visual objects, attributes, relations, counts, or text
that are NOT supported by the image or ground truth.

Examples:
- Image has no dog → model says "Yes, there is a dog"
- Image shows 3 apples → model says "There are five apples"
- Car is red → model says "The car is blue"

### 2. Factual Inconsistency
The model's answer contradicts the ground truth, question constraints,
or well-established facts referenced in the answer.

Examples:
- Chart shows highest in 2020 → model says 2019
- Given radius=5 in question → model uses radius=6
- Image shows 3 objects → model says 4

### 3. Reasoning Hallucination
The model introduces unsupported assumptions, wrong intermediate steps,
or fabricated evidence in its reasoning, even if the final answer
happens to be correct.

Examples:
- Model assumes angle=60° without image evidence
- Model invents "the diagram says AB=AC" when it doesn't
- Model uses wrong formula despite reading numbers correctly

## Special Rules

1. If the answer is WRONG but does NOT fabricate unsupported content,
   label as: hallucination=false, and note "wrong but not hallucination"
   in human_notes.

2. If the model fabricates visual evidence in the reasoning, label as
   hallucination=true even if the final answer is correct.

3. Format errors alone (e.g. missing units, slightly different phrasing)
   do NOT count as hallucination.

4. When multiple hallucination types apply, choose the PRIMARY error source.

5. If unsure, lower your confidence score and explain in human_notes.

## Confidence Scale
- 0.9-1.0: Clearly hallucination / clearly not, unambiguous
- 0.7-0.9: Likely hallucination / likely not, minor ambiguity
- 0.5-0.7: Uncertain, but leaning one way
- <0.5: Highly uncertain
