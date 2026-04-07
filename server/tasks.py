from typing import Dict, Any, Tuple
from models import Email
from datetime import datetime, timedelta

class EmailTask:
    def __init__(self, email: Email, correct_category: str, correct_priority: int, 
                 correct_action: str, difficulty: str, description: str,
                 time_limit: int = 300):
        self.email = email
        self.correct_category = correct_category
        self.correct_priority = correct_priority
        self.correct_action = correct_action
        self.difficulty = difficulty
        self.description = description
        self.time_limit = time_limit
        self.previous_score = None
    
    def grade(self, category: str, priority: int, action: str, 
              response: str = None, time_taken: float = 0.0,
              confidence: float = 1.0) -> Tuple[float, Dict[str, float]]:
        scores = {}
        
        category_score = 1.0 if category == self.correct_category else 0.0
        scores['category'] = category_score
        
        priority_diff = abs(priority - self.correct_priority)
        if priority_diff == 0:
            priority_score = 1.0
        elif priority_diff == 1:
            priority_score = 0.7
        elif priority_diff == 2:
            priority_score = 0.4
        else:
            priority_score = 0.1
        scores['priority'] = priority_score
        
        action_score = 1.0 if action == self.correct_action else 0.0
        scores['action'] = action_score
        
        response_score = 0.0
        if action == "reply":
            if response:
                word_count = len(response.split())
                length_score = min(1.0, word_count / 30.0)
                response_score = length_score
        else:
            response_score = 1.0
        scores['response'] = response_score
        
        time_penalty = 0.0
        if time_taken > 0:
            if time_taken > self.time_limit:
                time_penalty = -0.15
            else:
                time_penalty = -0.15 * (time_taken / self.time_limit)
        scores['time_penalty'] = time_penalty
        
        confidence_bonus = 0.0
        if category_score == 1.0 and priority_score >= 0.7 and action_score == 1.0:
            confidence_bonus = confidence * 0.10
        scores['confidence_bonus'] = confidence_bonus
        
        improvement_bonus = 0.0
        if self.previous_score is not None:
            improvement = self.previous_score - (category_score * 0.3 + priority_score * 0.25 + 
                                                  action_score * 0.2 + response_score * 0.15)
            if improvement > 0:
                improvement_bonus = min(0.05, improvement * 0.5)
        scores['improvement_bonus'] = improvement_bonus
        
        base_total = (
            scores['category'] * 0.30 +
            scores['priority'] * 0.25 +
            scores['action'] * 0.20 +
            scores['response'] * 0.15
        )
        
        total = base_total + time_penalty + confidence_bonus + improvement_bonus
        total = max(0.0, min(1.0, total))
        
        self.previous_score = total
        return total, scores


def create_easy_task() -> EmailTask:
    """EASY: Production server down - Critical, needs immediate action"""
    email = Email(
        email_id="easy_001",
        sender="alerts@monitoring.com",
        subject="[CRITICAL] Production Server Down - Immediate Action Required",
        body="""
ALERT: Production server api-01.prod.internal is DOWN.

Impact: Customer transactions failing (503 errors increasing)
Duration: 12 minutes
Priority: P0 - Critical

Please investigate and resolve immediately.

- Incident ID: INC-2024-042
- Affected Services: payment-api, checkout-service
""",
        timestamp=datetime.now() - timedelta(minutes=12),
        deadline=datetime.now() + timedelta(minutes=30),
        attachments=["error_logs.txt"],
        urgency_keywords=["critical", "down", "immediate", "p0"]
    )
    
    return EmailTask(
        email=email,
        correct_category="urgent",
        correct_priority=1,
        correct_action="reply",
        difficulty="easy",
        description="Production server is down - immediate response required (30 min deadline)",
        time_limit=120
    )


def create_medium_task() -> EmailTask:
    """MEDIUM: Client follow-up - Important but not urgent, needs professional response"""
    email = Email(
        email_id="medium_001",
        sender="sarah.johnson@clientcorp.com",
        subject="Q4 Marketing Campaign - Timeline Confirmation",
        body="""
Hi Team,

Following up on our call last week about the Q4 marketing campaign.

Could you please share the updated timeline for the social media deliverables?
We need to align this with our product launch on Dec 15th.

Also, can you confirm if the creative assets will be ready by Nov 30th?

Not super urgent, but would appreciate an update by Friday.

Best regards,
Sarah Johnson
Marketing Director, ClientCorp
""",
        timestamp=datetime.now() - timedelta(days=2),
        deadline=datetime.now() + timedelta(days=3),
        cc=["projectmanager@company.com"],
        urgency_keywords=["timeline", "deliverables", "by friday"]
    )
    
    return EmailTask(
        email=email,
        correct_category="important",
        correct_priority=3,
        correct_action="reply",
        difficulty="medium",
        description="Client follow-up on campaign timeline - respond within 3 days",
        time_limit=300
    )


def create_hard_task() -> EmailTask:
    """HARD: Board meeting in 2 hours - Urgent, confidential, requires analysis"""
    thread_history = [
        Email(
            email_id="thread_001",
            sender="finance@company.com",
            subject="Q4 Financial Review - Draft Numbers",
            body="Attached are the preliminary Q4 numbers. Revenue down 12% due to market conditions.",
            timestamp=datetime.now() - timedelta(days=3),
            attachments=["q4_draft.xlsx"]
        ),
        Email(
            email_id="thread_002",
            sender="legal@company.com",
            subject="Disclosure Requirements - Board Meeting",
            body="Legal review confirms we must disclose the 12% revenue decline at the board meeting.",
            timestamp=datetime.now() - timedelta(days=2),
            confidential=True
        )
    ]
    
    current_email = Email(
        email_id="hard_001",
        sender="board@company.com",
        subject="URGENT: Board Meeting in 2 Hours - Final Numbers Required",
        body="""
CONFIDENTIAL - For internal use only

Board meeting starts at 2 PM. We need:
1. Final Q4 numbers with analysis
2. Action plan for revenue recovery
3. Legal disclosure statement

The board expects a detailed presentation.

Please prepare immediately. This is time-sensitive.

- Board Secretary
""",
        timestamp=datetime.now() - timedelta(hours=1),
        thread_id="q4_review",
        previous_emails=thread_history,
        deadline=datetime.now() + timedelta(hours=2),
        confidential=True,
        attachments=["board_presentation_template.pptx", "q3_comparison.pdf"],
        cc=["ceo@company.com", "cfo@company.com"],
        urgency_keywords=["urgent", "board", "confidential", "2 hours"]
    )
    
    return EmailTask(
        email=current_email,
        correct_category="urgent",
        correct_priority=1,
        correct_action="reply",
        difficulty="hard",
        description="Board meeting in 2 hours - urgent response with confidential analysis",
        time_limit=600
    )


def get_all_tasks() -> Dict[str, EmailTask]:
    return {
        "easy": create_easy_task(),
        "medium": create_medium_task(),
        "hard": create_hard_task()
    }