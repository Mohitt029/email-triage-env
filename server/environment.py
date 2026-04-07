from typing import Optional
import uuid
import time
from datetime import datetime

from models import (
    EmailTriageAction, EmailTriageObservation, EmailTriageState,
    Email, GraderResult
)
from server.tasks import get_all_tasks, EmailTask

class EmailTriageEnvironment:
    def __init__(self):
        self.tasks = get_all_tasks()
        self.current_task: Optional[EmailTask] = None
        self.current_difficulty: Optional[str] = None
        self.episode_id: Optional[str] = None
        self.step_count = 0
        self.total_reward = 0.0
        self.decisions = []
        self.done = False
        self.episode_start_time = None
        
    def reset(self) -> EmailTriageObservation:
        # FIX: Use the current difficulty if set, otherwise default to easy
        if self.current_difficulty is None:
            self.current_difficulty = "easy"
        
        self.current_task = self.tasks[self.current_difficulty]
        self.episode_id = str(uuid.uuid4())
        self.step_count = 0
        self.total_reward = 0.0
        self.decisions = []
        self.done = False
        self.episode_start_time = time.time()
        
        return EmailTriageObservation(
            current_email=self.current_task.email,
            task_description=self.current_task.description,
            task_difficulty=self.current_difficulty,
            previous_decisions=[],
            emails_processed=0,
            total_emails=1,
            time_elapsed=0.0,
            time_remaining=self.current_task.time_limit,
            done=False,
            reward=0.0
        )
    
    def step(self, action: EmailTriageAction, time_taken: float = 0.0, confidence: float = 1.0) -> EmailTriageObservation:
        if self.done:
            raise RuntimeError("Episode already done. Call reset() first.")
        
        if time_taken == 0.0 and self.episode_start_time:
            time_taken = time.time() - self.episode_start_time
        
        score, breakdown = self.current_task.grade(
            category=action.category,
            priority=action.priority,
            action=action.action,
            response=action.response_text,
            time_taken=time_taken,
            confidence=confidence
        )
        
        self.step_count += 1
        self.total_reward += score
        
        self.decisions.append({
            "step": self.step_count,
            "action": action.__dict__,
            "score": score,
            "breakdown": breakdown,
            "time_taken": time_taken,
            "confidence": confidence
        })
        
        self.done = True
        
        time_remaining = max(0, self.current_task.time_limit - time_taken)
        
        return EmailTriageObservation(
            current_email=self.current_task.email,
            task_description=self.current_task.description,
            task_difficulty=self.current_difficulty,
            previous_decisions=self.decisions,
            emails_processed=1,
            total_emails=1,
            time_elapsed=time_taken,
            time_remaining=time_remaining,
            done=self.done,
            reward=score
        )
    
    @property
    def state(self) -> EmailTriageState:
        avg_time = 0.0
        if self.decisions:
            avg_time = sum(d.get('time_taken', 0) for d in self.decisions) / len(self.decisions)
        
        return EmailTriageState(
            episode_id=self.episode_id or "not_started",
            step_count=self.step_count,
            current_task_id=hash(self.current_task) if self.current_task else 0,
            task_difficulty=self.current_difficulty or "unknown",
            total_reward=self.total_reward,
            correct_decisions=sum(1 for d in self.decisions if d['score'] > 0.8),
            total_decisions=len(self.decisions),
            email_queue=[],
            total_time_spent=sum(d.get('time_taken', 0) for d in self.decisions),
            average_response_time=avg_time,
            improvement_rate=self._calculate_improvement()
        )
    
    def _calculate_improvement(self) -> float:
        if len(self.decisions) < 2:
            return 0.0
        scores = [d['score'] for d in self.decisions]
        return (scores[-1] - scores[0]) / len(scores)
    
    def get_grader_result(self) -> GraderResult:
        if not self.decisions:
            return GraderResult(
                score=0.0,
                feedback="No decisions made",
                breakdown={}
            )
        
        avg_score = sum(d['score'] for d in self.decisions) / len(self.decisions)
        avg_time = sum(d.get('time_taken', 0) for d in self.decisions) / len(self.decisions)
        
        if avg_score >= 0.9:
            feedback = f"Excellent! Perfect handling in {avg_time:.1f}s."
        elif avg_score >= 0.7:
            feedback = f"Good. Minor improvements needed. Response time: {avg_time:.1f}s"
        elif avg_score >= 0.5:
            feedback = f"Fair. Review priority and action selection."
        else:
            feedback = f"Needs improvement. Check task requirements."
        
        last = self.decisions[-1]
        breakdown = last.get('breakdown', {})
        
        return GraderResult(
            score=avg_score,
            feedback=feedback,
            breakdown=breakdown,
            time_penalty=breakdown.get('time_penalty', 0.0),
            confidence_bonus=breakdown.get('confidence_bonus', 0.0),
            improvement_bonus=breakdown.get('improvement_bonus', 0.0)
        )
    
    def set_task_difficulty(self, difficulty: str):
        """Set the task difficulty for the NEXT reset"""
        if difficulty in self.tasks:
            self.current_difficulty = difficulty
            
        else:
            raise ValueError(f"Unknown difficulty: {difficulty}. Options: {list(self.tasks.keys())}")
