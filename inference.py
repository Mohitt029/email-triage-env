"""
Inference Script for Email Triage Environment
"""
import os
import sys
import json
import argparse
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
            print(f"Using OpenAI - Model: {MODEL_NAME}")
        else:
            print("Using mock agent")
    
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
            print(f"API error: {e}")
            return self._mock_decision(observation)
    
    def _mock_decision(self, observation) -> EmailTriageAction:
        body = observation.current_email.body.lower()
        if any(k in body for k in ["urgent", "asap", "critical", "down"]):
            return EmailTriageAction(category="urgent", priority=1, action="reply", response_text="Investigating", confidence=0.9)
        elif any(k in body for k in ["important", "client", "meeting"]):
            return EmailTriageAction(category="important", priority=2, action="reply", response_text="Will review", confidence=0.8)
        return EmailTriageAction(category="normal", priority=3, action="archive", confidence=0.5)


def run_task(env: EmailTriageEnv, task_id: str, agent: BaselineAgent) -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"Running {task_id.upper()} task")
    print('='*60)
    
    try:
        obs = env.reset()
        print(f"Email: {obs.current_email.subject}")
        
        action = agent.decide(obs)
        print(f"Decision: {action.category}, P{action.priority}, {action.action}")
        
        result = env.step(action)
        score = env.get_grader_score()
        print(f"Score: {score:.3f}")
        
        return {"task": task_id, "score": score}
        
    except Exception as e:
        print(f"Error on {task_id}: {e}")
        return {"task": task_id, "score": 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="https://MohitS029-email-triage-env1.hf.space", help="Environment URL")
    parser.add_argument("--eval-only", action="store_true", help="Output JSON")
    args = parser.parse_args()
    
    env = EmailTriageEnv(base_url=args.url)
    agent = BaselineAgent()
    
    results = []
    for task in ["easy", "medium", "hard"]:
        results.append(run_task(env, task, agent))
    
    scores = {r["task"]: r["score"] for r in results}
    avg = sum(scores.values()) / 3 if scores else 0.0
    
    if args.eval_only:
        print(json.dumps({"scores": scores, "average": avg}, indent=2))
    else:
        print(f"\n{'='*60}")
        print("BASELINE RESULTS")
        print('='*60)
        for task, score in scores.items():
            print(f"{task.upper()}: {score:.3f}")
        print(f"\nAVERAGE: {avg:.3f}")
    
    env.close()

if __name__ == "__main__":
    main()