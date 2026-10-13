# AI6130: Assignment 1 
# Benchmark Evaluation with Open-LLM
## Submission Details
●	Due Date: 14 October 2026 (11:59 PM)
●	Submission: only the report (don’t need to submit code) to NTULearn (do not submit via email):
○	Link: Assignments → Assignment 1 → Report
○	Name the submission file: "AI6130-Ass1-YourStudentID"
## Objective
In this assignment, you will evaluate LLMs on the Open-LLM-Benchmark using the Hugging Face framework.
## Example Codebase
You are provided with an example implementation:
https://colab.research.google.com/drive/1DB9pOMznWuBFHgb8YBRR2Ctbil3l-Qma?usp=sharing
The notebook demonstrates how to:
●	Load the Open-Style/Open-LLM-Benchmark dataset from Hugging Face.
●	Load an LLM for inference on the benchmark tasks.
## Tasks
1. Warm-up
●	Read and understand the provided example code.
●	Run the notebook to verify your setup.
2. Benchmark Evaluation
●	Load three tasks from the Open-LLM-Benchmark:
○	CommonsenseQA
○	OpenbookQA
○	PiQA
●	Load four LLMs:
○	TinyLlama_v1.1: TinyLlama/TinyLlama_v1.1
○	Qwen/Qwen2.5-3B-Instruct: Qwen/Qwen2.5-3B-Instruct
○	DeepSeek-R1-Distill-Qwen-1.5B: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
○	Qwen3-4B-Instruct-2507: Qwen/Qwen3-4B-Instruct-2507

●	Run each model on all three tasks and record the evaluation results.
3. Prompt Engineering
●	Modify the prompt used in the infer_llm function to improve inference, and ensure that it is compatible with the model template (check the Hugging Face model page if available).
●	Example modifications:
○	Original Prompt: "Question … Answer"
○	Improved Prompt: "Question … Answer in A/B/C/D"
●	You may also implement few-shot prompting (e.g., including multiple examples in the prompt).
●	Run each model again on the three tasks with your improved prompts and record the evaluation results.
Deliverables
Report: 
•	Summarize the results of experiments (preferably in tables)
•	Discuss your method of prompt modifications, conduct analysis, comparison, and explanation about the results 
•	The format is free style. Try to be concise and not more than 5 pages (including the cover page)
---------------------------------------------------------------------
Some useful links:
1.	Dataset:
Open LLM dataset: Open-Style/Open-LLM-Benchmark
Open LLM Leaderboard: open-llm-leaderboard/open_llm_leaderboard
2.	Model usage:
	For example: microsoft/phi-2
	How to load the model using Hugging Face:
a.	Visit the model page: (https://huggingface.co/ + model name)
 
b.	Click these buttons:
 
c.	Copy from the code block using AutoTokenizer and AutoModelForCasualLM into your Google Colab to use the model:
 
Note: Certain models may provide a predefined chat template, which you should follow if available.