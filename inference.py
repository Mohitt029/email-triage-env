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

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

class BaselineAgent:
    def __init__(self):
        self.use_api = API_BASE_URL and MODEL_NAME and HF_TOKEN
        if self.use_api:
            self.client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
            print(f"Using OpenAI - Model: {MODEL_NAME}", flush=True)
        else:
            print("Using mock agent", flush=True)
    
    def decide(self, observation) -> EmailTriageAction:
        if self.use_api:
            return self._api_decision(observation)
        return self._mock_decision(observation)
    
    def _api_decision(self, observation) -> EmailTriageAction:
        email = observation.current_email
        prompt = f"""Analyze this email.
From: {email.sender}
Subject: {email.subject}
Body: {email.body}
Task: {observation.task_description}
Respond with JSON: {{"category":"urgent/important/normal/spam","priority":1-5,"action":"reply/archive/delegate/flag/delete","response_text":"if reply","confidence":0.0-1.0}}"""
        
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Email triage assistant. Respond with JSON only."},
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
        except Exception as e:
            print(f"API error: {e}", flush=True)
            return self._mock_decision(observation)
    
    def _mock_decision(self, observation) -> EmailTriageAction:
        body = observation.current_email.body.lower()
        if any(k in body for k in ["urgent", "asap", "critical", "down", "server"]):
            return EmailTriageAction(category="urgent", priority=1, action="reply", response_text="Investigating immediately", confidence=0.9)
        elif any(k in body for k in ["important", "client", "meeting", "timeline"]):
            return EmailTriageAction(category="important", priority=2, action="reply", response_text="Will review and respond", confidence=0.8)
        return EmailTriageAction(category="normal", priority=3, action="archive", confidence=0.5)


def run_task(task_name: str, agent: BaselineAgent, space_url: str) -> Dict[str, Any]:
    """Run a single task with fresh environment"""
    
    print(f"[START] task={task_name}", flush=True)
    
    # Create a completely new environment for each attempt
    for attempt in range(3):
        env = None
        try:
            env = EmailTriageEnv(base_url=space_url)
            obs = env.reset()
            print(f"[STEP] step=1 reward=0.0", flush=True)
            
            time.sleep(0.5)
            
            action = agent.decide(obs)
            result = env.step(action)
            
            time.sleep(0.5)
            
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
    
    agent = BaselineAgent()
    
    results = []
    # Run easy, hard, medium order (medium last to avoid session issues)
    for task in ["easy", "hard", "medium"]:
        results.append(run_task(task, agent, args.url))
        time.sleep(1)
    
    # Reorder results for output
    scores_dict = {}
    for r in results:
        scores_dict[r["task"]] = r["score"]
    
    # Ensure all three tasks have scores
    scores = {
        "easy": scores_dict.get("easy", 0.0),
        "medium": scores_dict.get("medium", 0.0),
        "hard": scores_dict.get("hard", 0.0)
    }
    avg = sum(scores.values()) / 3
    
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