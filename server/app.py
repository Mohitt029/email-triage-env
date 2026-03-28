import sys
import os
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
import uuid
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from server.environment import EmailTriageEnvironment
from models import GraderResult, EmailTriageAction, EmailTriageObservation

app = FastAPI(
    title="Email Triage Environment",
    description="Real-world email triage environment for RL agents",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions = {}
metrics = {
    "total_episodes": 0,
    "total_scores": {"easy": [], "medium": [], "hard": []},
    "average_response_times": {"easy": [], "medium": [], "hard": []},
    "common_errors": {"category": 0, "priority": 0, "action": 0},
    "improvement_rate": 0.0
}

def get_or_create_session(session_id: Optional[str] = None):
    if session_id and session_id in sessions:
        return sessions[session_id], session_id
    new_session_id = str(uuid.uuid4())
    sessions[new_session_id] = EmailTriageEnvironment()
    metrics["total_episodes"] += 1
    return sessions[new_session_id], new_session_id


@app.post("/reset")
async def reset(session_id: Optional[str] = Query(None)):
    env, sid = get_or_create_session(session_id)
    start_time = time.time()
    observation = env.reset()
    
    if sid not in metrics:
        metrics[sid] = {}
    metrics[sid]["start_time"] = start_time
    metrics[sid]["task_difficulty"] = observation.task_difficulty
    
    return {
        "session_id": sid,
        "observation": observation.__dict__
    }


@app.post("/step")
async def step(
    action: EmailTriageAction,
    session_id: Optional[str] = Query(None)
):
    if not session_id or session_id not in sessions:
        raise HTTPException(status_code=400, detail="No active session. Call reset first.")
    
    env = sessions[session_id]
    start_time = metrics.get(session_id, {}).get("start_time", time.time())
    time_taken = time.time() - start_time
    
    result = env.step(action, time_taken=time_taken, confidence=action.confidence)
    
    task_difficulty = metrics.get(session_id, {}).get("task_difficulty", "easy")
    metrics["average_response_times"][task_difficulty].append(time_taken)
    
    if result.reward < 0.5:
        if action.category != env.current_task.correct_category:
            metrics["common_errors"]["category"] += 1
        if action.priority != env.current_task.correct_priority:
            metrics["common_errors"]["priority"] += 1
        if action.action != env.current_task.correct_action:
            metrics["common_errors"]["action"] += 1
    
    return {
        "session_id": session_id,
        "observation": result.__dict__,
        "reward": result.reward,
        "done": result.done,
        "time_taken": time_taken
    }


@app.get("/state")
async def get_state(session_id: str = Query(...)):
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Session not found")
    env = sessions[session_id]
    return env.state.__dict__


@app.get("/tasks")
async def get_tasks() -> Dict[str, Any]:
    from server.tasks import get_all_tasks
    tasks = get_all_tasks()
    
    return {
        "tasks": [
            {
                "id": difficulty,
                "description": task.description,
                "difficulty": difficulty,
                "expected_category": task.correct_category,
                "expected_priority": task.correct_priority,
                "expected_action": task.correct_action,
                "time_limit": task.time_limit
            }
            for difficulty, task in tasks.items()
        ],
        "action_schema": {
            "category": {"type": "string", "enum": ["urgent", "important", "normal", "spam"], "description": "Email category"},
            "priority": {"type": "integer", "minimum": 1, "maximum": 5, "description": "Priority level (1=highest)"},
            "action": {"type": "string", "enum": ["reply", "archive", "delegate", "flag", "delete"], "description": "Action to take"},
            "response_text": {"type": "string", "description": "Required if action is 'reply'"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1, "description": "Agent's confidence"},
            "reasoning": {"type": "string", "description": "Reasoning behind the decision"}
        }
    }


@app.post("/grader")
async def get_grader(session_id: str = Query(...)):
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Session not found")
    
    env = sessions[session_id]
    if env.state.step_count == 0:
        raise HTTPException(status_code=400, detail="No episode completed yet")
    
    result = env.get_grader_result()
    
    task_difficulty = metrics.get(session_id, {}).get("task_difficulty", "easy")
    metrics["total_scores"][task_difficulty].append(result.score)
    
    if len(metrics["total_scores"][task_difficulty]) > 1:
        recent_scores = metrics["total_scores"][task_difficulty][-5:]
        metrics["improvement_rate"] = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
    
    score = max(0.0, min(1.0, result.score))
    
    return {
        "score": score,
        "feedback": result.feedback,
        "breakdown": result.breakdown,
        "steps_taken": env.state.step_count,
        "total_reward": env.state.total_reward,
        "time_penalty": getattr(result, 'time_penalty', 0.0),
        "confidence_bonus": getattr(result, 'confidence_bonus', 0.0),
        "improvement_bonus": getattr(result, 'improvement_bonus', 0.0)
    }


@app.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    avg_scores = {}
    avg_response_times = {}
    
    for diff in ["easy", "medium", "hard"]:
        scores = metrics["total_scores"][diff]
        avg_scores[diff] = sum(scores) / len(scores) if scores else 0.0
        
        times = metrics["average_response_times"][diff]
        avg_response_times[diff] = sum(times) / len(times) if times else 0.0
    
    return {
        "total_episodes": metrics["total_episodes"],
        "average_scores": avg_scores,
        "average_response_times": avg_response_times,
        "common_errors": metrics["common_errors"],
        "improvement_rate": metrics["improvement_rate"],
        "tasks_completed": {
            "easy": len(metrics["total_scores"]["easy"]),
            "medium": len(metrics["total_scores"]["medium"]),
            "hard": len(metrics["total_scores"]["hard"])
        }
    }


@app.post("/baseline")
async def run_baseline(request: Request) -> Dict[str, Any]:
    import subprocess
    import json
    
    try:
        result = subprocess.run(
            [sys.executable, "inference.py", "--eval-only"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=os.path.dirname(os.path.dirname(__file__))
        )
        
        if result.returncode != 0:
            return {"error": "Baseline script failed", "stderr": result.stderr, "status": "failed"}
        
        try:
            output_data = json.loads(result.stdout)
            scores = output_data.get("scores", {})
        except:
            import re
            scores = {}
            for diff in ["easy", "medium", "hard"]:
                match = re.search(f"{diff.upper()}: (\\d+\\.?\\d*)", result.stdout)
                if match:
                    scores[diff] = float(match.group(1))
                else:
                    scores[diff] = 0.0
        
        return {
            "status": "success",
            "scores": scores,
            "average": sum(scores.values()) / len(scores) if scores else 0.0,
            "output": result.stdout
        }
        
    except subprocess.TimeoutExpired:
        return {"error": "Baseline script timed out", "status": "failed"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "healthy", "environment": "email_triage", "version": "2.0.0"}


@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "name": "Email Triage Environment",
        "version": "2.0.0",
        "description": "Real-world email triage environment for RL agents",
        "endpoints": {
            "/reset": "POST - Start new episode",
            "/step": "POST - Take action",
            "/state": "GET - Get current state",
            "/tasks": "GET - List tasks",
            "/grader": "POST - Get episode score",
            "/metrics": "GET - Performance dashboard",
            "/baseline": "POST - Run baseline",
            "/health": "GET - Health check"
        }
    }


@app.post("/select_task/{difficulty}")
async def select_task(difficulty: str, session_id: str = Query(...)):
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Session not found")
    
    env = sessions[session_id]
    if difficulty not in ["easy", "medium", "hard"]:
        raise HTTPException(status_code=400, detail="Invalid difficulty")
    
    env.set_task_difficulty(difficulty)
    
    if session_id in metrics:
        metrics[session_id]["task_difficulty"] = difficulty
    
    return {
        "session_id": session_id,
        "difficulty": difficulty,
        "task_description": env.current_task.description,
        "email_subject": env.current_task.email.subject,
        "email_sender": env.current_task.email.sender,
        "time_limit": env.current_task.time_limit,
        "confidential": env.current_task.email.confidential
    }