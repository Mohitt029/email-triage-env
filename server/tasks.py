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
        
        # Category - 30%
        category_score = 1.0 if category == self.correct_category else 0.0
        scores['category'] = category_score
        
        # Priority - 25%
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
        
        # Action - 20%
        action_score = 1.0 if action == self.correct_action else 0.0
        scores['action'] = action_score
        
        # Response - 15%
        response_score = 0.0
        if action == "reply":
            if response:
                word_count = len(response.split())
                length_score = min(1.0, word_count / 30.0)
                response_score = length_score
        else:
            response_score = 1.0
        scores['response'] = response_score
        
        # Time penalty - up to -15%
        time_penalty = 0.0
        if time_taken > 0:
            if time_taken > self.time_limit:
                time_penalty = -0.15
            else:
                time_penalty = -0.15 * (time_taken / self.time_limit)
        scores['time_penalty'] = time_penalty
        
        # Confidence bonus - up to +10%
        confidence_bonus = 0.0
        if category_score == 1.0 and priority_score >= 0.7 and action_score == 1.0:
            confidence_bonus = confidence * 0.10
        scores['confidence_bonus'] = confidence_bonus
        
        # Improvement bonus - up to +5%
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
    email = Email(
        email_id="easy_001",
        sender="urgent@support.com",
        subject="URGENT: Server Down - Production Impact",
        body="""
        Hi Team,
        
        Our main production server is down. Customers are unable to access 
        the application. This is a critical issue that needs immediate attention.
        
        Please investigate ASAP.
        
        - Support Team
        """,
        timestamp=datetime.now() - timedelta(hours=1),
        deadline=datetime.now() + timedelta(minutes=30),
        attachments=["error_logs.txt"]
    )
    
    return EmailTask(
        email=email,
        correct_category="urgent",
        correct_priority=1,
        correct_action="reply",
        difficulty="easy",
        description="Clear urgent production issue requiring immediate response (30 min deadline)",
        time_limit=120
    )

def create_medium_task() -> EmailTask:
    email = Email(
        email_id="medium_001",
        sender="client@company.com",
        subject="Question about our recent meeting",
        body="""
        Hi,
        
        Following up on our meeting yesterday about the Q3 deliverables.
        I wanted to confirm if we're still on track for the end of month?
        Also, could you send over the updated timeline when you get a chance?
        
        Thanks,
        Client
        """,
        timestamp=datetime.now() - timedelta(days=2),
        deadline=datetime.now() + timedelta(days=1),
        cc=["manager@company.com"]
    )
    
    return EmailTask(
        email=email,
        correct_category="important",
        correct_priority=3,
        correct_action="reply",
        difficulty="medium",
        description="Client follow-up requiring thoughtful response (24 hour deadline)",
        time_limit=300
    )

def create_hard_task() -> EmailTask:
    thread_history = [
        Email(
            email_id="thread_001",
            sender="manager@company.com",
            subject="RE: Q4 Budget Review - Urgent",
            body="We need the Q4 budget numbers by EOD today for the board meeting.",
            timestamp=datetime.now() - timedelta(days=1)
        ),
        Email(
            email_id="thread_002", 
            sender="finance@company.com",
            subject="RE: Q4 Budget Review - Urgent",
            body="Waiting on numbers from sales. Will send as soon as I get them.",
            timestamp=datetime.now() - timedelta(hours=12)
        )
    ]
    
    current_email = Email(
        email_id="hard_001",
        sender="ceo@company.com",
        subject="RE: Q4 Budget Review - CRITICAL - Board in 2 Hours",
        body="""
        Team,
        
        The board meeting is in 2 hours and we still don't have the budget numbers.
        Legal says we must disclose the 15% decline today.
        
        I need updated numbers from sales and your analysis.
        
        This is confidential.
        """,
        timestamp=datetime.now() - timedelta(hours=2),
        thread_id="budget_thread",
        previous_emails=thread_history,
        deadline=datetime.now() + timedelta(minutes=120),
        confidential=True,
        attachments=["q4_preliminary.pdf", "legal_memo_2024.pdf"],
        cc=["board@company.com", "legal@company.com"]
    )
    
    return EmailTask(
        email=current_email,
        correct_category="urgent",
        correct_priority=1,
        correct_action="reply",
        difficulty="hard",
        description="Complex multi-email thread with confidentiality and 2-hour board deadline",
        time_limit=600
    )

def get_all_tasks() -> Dict[str, EmailTask]:
    return {
        "easy": create_easy_task(),
        "medium": create_medium_task(),
        "hard": create_hard_task()
    }