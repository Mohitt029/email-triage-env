"""
HTTP Client for Email Triage Environment
"""
import requests
from typing import Optional, Dict, Any
from models import EmailTriageAction, EmailTriageObservation, EmailTriageState, Email

class EmailTriageEnv:
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self._session = requests.Session()
        self._session_id = None
    
    def reset(self) -> EmailTriageObservation:
        response = self._session.post(
            f"{self.base_url}/reset",
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        
        # Store the session ID for future calls
        self._session_id = data.get('session_id')
        obs_data = data.get('observation', {})
        
        email_data = obs_data.get('current_email', {})
        current_email = Email(
            email_id=email_data.get('email_id', ''),
            sender=email_data.get('sender', ''),
            subject=email_data.get('subject', ''),
            body=email_data.get('body', ''),
            timestamp=email_data.get('timestamp', ''),
            attachments=email_data.get('attachments', []),
            confidential=email_data.get('confidential', False),
            deadline=email_data.get('deadline'),
            sentiment=email_data.get('sentiment', 'neutral')
        )
        
        return EmailTriageObservation(
            current_email=current_email,
            task_description=obs_data.get('task_description', ''),
            task_difficulty=obs_data.get('task_difficulty', ''),
            previous_decisions=obs_data.get('previous_decisions', []),
            emails_processed=obs_data.get('emails_processed', 0),
            total_emails=obs_data.get('total_emails', 1),
            time_elapsed=obs_data.get('time_elapsed', 0.0),
            time_remaining=obs_data.get('time_remaining', 300.0),
            done=obs_data.get('done', False),
            reward=obs_data.get('reward', 0.0)
        )
    
    def step(self, action: EmailTriageAction) -> EmailTriageObservation:
        if not self._session_id:
            raise RuntimeError("Must call reset() first")
        
        payload = {
            "category": action.category,
            "priority": action.priority,
            "action": action.action,
            "confidence": getattr(action, 'confidence', 1.0)
        }
        if action.response_text:
            payload["response_text"] = action.response_text
        
        response = self._session.post(
            f"{self.base_url}/step?session_id={self._session_id}",
            json=payload,
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            print(f"Error response: {response.text}")
            response.raise_for_status()
        
        data = response.json()
        
        obs_data = data.get('observation', {})
        email_data = obs_data.get('current_email', {})
        current_email = Email(
            email_id=email_data.get('email_id', ''),
            sender=email_data.get('sender', ''),
            subject=email_data.get('subject', ''),
            body=email_data.get('body', ''),
            timestamp=email_data.get('timestamp', ''),
            attachments=email_data.get('attachments', []),
            confidential=email_data.get('confidential', False),
            deadline=email_data.get('deadline'),
            sentiment=email_data.get('sentiment', 'neutral')
        )
        
        return EmailTriageObservation(
            current_email=current_email,
            task_description=obs_data.get('task_description', ''),
            task_difficulty=obs_data.get('task_difficulty', ''),
            previous_decisions=obs_data.get('previous_decisions', []),
            emails_processed=obs_data.get('emails_processed', 0),
            total_emails=obs_data.get('total_emails', 1),
            time_elapsed=obs_data.get('time_elapsed', 0.0),
            time_remaining=obs_data.get('time_remaining', 300.0),
            done=data.get('done', False),
            reward=data.get('reward', 0.0)
        )
    
    def get_grader_score(self) -> float:
        if not self._session_id:
            raise RuntimeError("Must call reset() first")
        
        response = self._session.post(
            f"{self.base_url}/grader?session_id={self._session_id}",
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        return data.get('score', 0.0)
    
    def get_tasks(self) -> Dict[str, Any]:
        response = self._session.get(
            f"{self.base_url}/tasks",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def close(self):
        self._session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()