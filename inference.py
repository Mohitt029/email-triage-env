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

# Environment variables - with fallback for local testing
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
API_KEY = os.environ.get("HF_TOKEN", "")

# Initialize OpenAI client only if API key is available
client = None
if API_KEY:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def decide(observation) -> EmailTriageAction:
    """Make decision using OpenAI API or fallback logic"""
    email = observation.current_email
    
    # If API client is available, use it
    if client:
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
    
    # Fallback logic based on email content
    body = email.body.lower()
    if any(k in body for k in ["urgent", "critical", "down", "server"]):
        return EmailTriageAction(category="urgent", priority=1, action="reply", response_text="Investigating", confidence=0.9)
    elif any(k in body for k in ["client", "meeting", "timeline"]):
        return EmailTriageAction(category="important", priority=2, action="reply", response_text="Will review", confidence=0.8)
    else:
        return EmailTriageAction(category="normal", priority=3, action="archive", confidence=0.7)


def run_task(task_name: str, space_url: str) -> Dict[str, Any]:
    """Run a single task with retry logic"""
    
    print(f"[START] task={task_name}", flush=True)
    
    for attempt in range(3):
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
            print(f"Attempt {attempt + 1} failed for {task_name}: {e}", flush=True)
            if attempt == 2:
                print(f"[END] task={task_name} score=0.0 steps=0", flush=True)
                return {"task": task_name, "score": 0.0}
            time.sleep(1)
        finally:
            if env:
                env.close()
            time.sleep(0.5)
    
    return {"task": task_name, "score": 0.0}


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