"""
Inference Script for Email Triage Environment
"""
import os
import sys
import json
import argparse
import time
from typing import Dict, Any
from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))

from client import EmailTriageEnv
from models import EmailTriageAction

# Environment variables - injected by evaluation system
API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.environ["MODEL_NAME"]
API_KEY = os.environ["HF_TOKEN"]

# OpenAI client for LLM calls
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def decide(observation) -> EmailTriageAction:
    email = observation.current_email
    
    prompt = f"""Analyze this email and make a triage decision.

Email:
From: {email.sender}
Subject: {email.subject}
Body: {email.body}
Task: {observation.task_description}

Respond with JSON only in this exact format:
{{"category":"urgent/important/normal/spam","priority":1-5,"action":"reply/archive/delegate/flag/delete","response_text":"if reply","confidence":0.0-1.0}}"""
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an email triage assistant. Respond with JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=200
    )
    
    result = json.loads(response.choices[0].message.content)
    
    return EmailTriageAction(
        category=result.get("category", "normal"),
        priority=int(result.get("priority", 3)),
        action=result.get("action", "archive"),
        response_text=result.get("response_text"),
        confidence=float(result.get("confidence", 0.8)),
        reasoning="API decision"
    )


def run_task(task_name: str, space_url: str) -> Dict[str, Any]:
    print(f"[START] task={task_name}", flush=True)
    env = None
    
    try:
        env = EmailTriageEnv(base_url=space_url)
        obs = env.reset()
        print(f"[STEP] step=1 reward=0.0", flush=True)
        
        time.sleep(0.3)
        action = decide(obs)
        result = env.step(action)
        time.sleep(0.3)
        
        score = env.get_grader_score()
        print(f"[END] task={task_name} score={score:.3f} steps=1", flush=True)
        return {"task": task_name, "score": score}
        
    except Exception as e:
        print(f"Error on {task_name}: {e}", flush=True)
        print(f"[END] task={task_name} score=0.0 steps=0", flush=True)
        return {"task": task_name, "score": 0.0}
    
    finally:
        if env:
            env.close()
        time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="https://MohitS029-email-triage-env1.hf.space", help="Environment URL")
    parser.add_argument("--eval-only", action="store_true", help="Output JSON")
    args = parser.parse_args()
    
    results = []
    for task in ["easy", "medium", "hard"]:
        results.append(run_task(task, args.url))
        time.sleep(0.5)
    
    scores = {r["task"]: r["score"] for r in results}
    avg = sum(scores.values()) / 3 if scores else 0.0
    
    if args.eval_only:
        print(json.dumps({"scores": scores, "average": avg}, indent=2), flush=True)
    
    print(f"\n{'='*50}", flush=True)
    print("BASELINE RESULTS", flush=True)
    print('='*50, flush=True)
    for task, score in scores.items():
        print(f"{task.upper()}: {score:.3f}", flush=True)
    print(f"\nAVERAGE: {avg:.3f}", flush=True)


if __name__ == "__main__":
    main()