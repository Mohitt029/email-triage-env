from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime, timedelta

class EmailCategory(Enum):
    URGENT = "urgent"
    IMPORTANT = "important"
    NORMAL = "normal"
    SPAM = "spam"
    UNKNOWN = "unknown"

class ActionType(Enum):
    REPLY = "reply"
    ARCHIVE = "archive"
    DELEGATE = "delegate"
    FLAG = "flag"
    DELETE = "delete"

class Priority(Enum):
    P1 = 1
    P2 = 2
    P3 = 3
    P4 = 4
    P5 = 5

class Sentiment(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    URGENT_NEGATIVE = "urgent_negative"


@dataclass
class EmailTriageAction:
    """What the AI sends to the environment"""
    category: str
    priority: int
    action: str
    response_text: Optional[str] = None
    confidence: float = 1.0
    reasoning: Optional[str] = None
    
    def __post_init__(self):
        valid_categories = ["urgent", "important", "normal", "spam"]
        if self.category not in valid_categories:
            raise ValueError(f"category must be one of {valid_categories}")
        
        if not 1 <= self.priority <= 5:
            raise ValueError("priority must be between 1 and 5")
        
        valid_actions = ["reply", "archive", "delegate", "flag", "delete"]
        if self.action not in valid_actions:
            raise ValueError(f"action must be one of {valid_actions}")
        
        if self.action == "reply" and not self.response_text:
            raise ValueError("response_text required when action is 'reply'")
        
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")


@dataclass
class Email:
    """Email structure with metadata like attachments, deadlines, confidentiality"""
    email_id: str
    sender: str
    subject: str
    body: str
    timestamp: datetime
    thread_id: Optional[str] = None
    previous_emails: List['Email'] = None
    attachments: List[str] = field(default_factory=list)
    cc: List[str] = field(default_factory=list)
    bcc: List[str] = field(default_factory=list)
    confidential: bool = False
    deadline: Optional[datetime] = None
    sentiment: Optional[str] = None
    urgency_keywords: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.sentiment is None:
            self.sentiment = self._detect_sentiment()
        if not self.urgency_keywords:
            self.urgency_keywords = self._detect_urgency_keywords()
    
    def _detect_sentiment(self) -> str:
        body_lower = self.body.lower()
        positive_words = ["thank", "appreciate", "great", "good", "pleased", "happy"]
        negative_words = ["urgent", "critical", "issue", "problem", "failed", "down", "error"]
        
        pos_count = sum(1 for w in positive_words if w in body_lower)
        neg_count = sum(1 for w in negative_words if w in body_lower)
        
        if neg_count > pos_count + 2:
            return "urgent_negative"
        elif neg_count > pos_count:
            return "negative"
        elif pos_count > neg_count:
            return "positive"
        return "neutral"
    
    def _detect_urgency_keywords(self) -> List[str]:
        body_lower = self.body.lower()
        urgency_indicators = ["urgent", "asap", "immediately", "critical", "emergency", 
                              "deadline", "today", "now", "important", "attention"]
        return [word for word in urgency_indicators if word in body_lower]
    
    def to_text(self) -> str:
        text = f"""
        From: {self.sender}
        Subject: {self.subject}
        Time: {self.timestamp.strftime('%Y-%m-%d %H:%M')}
        Sentiment: {self.sentiment}
        Confidential: {self.confidential}
        """
        if self.deadline:
            text += f"Deadline: {self.deadline.strftime('%Y-%m-%d %H:%M')}\n"
        if self.attachments:
            text += f"Attachments: {', '.join(self.attachments)}\n"
        text += f"\n{self.body}"
        return text


@dataclass
class EmailTriageObservation:
    """What the AI sees after each action"""
    current_email: Email
    task_description: str
    task_difficulty: str
    previous_decisions: List[Dict[str, Any]] = None
    emails_processed: int = 0
    total_emails: int = 1
    time_elapsed: float = 0.0
    time_remaining: float = 300.0
    done: bool = False
    reward: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EmailTriageState:
    """Keeps track of what happened during the episode"""
    episode_id: str
    step_count: int
    current_task_id: int
    task_difficulty: str
    total_reward: float
    correct_decisions: int
    total_decisions: int
    email_queue: List[str]
    total_time_spent: float = 0.0
    average_response_time: float = 0.0
    improvement_rate: float = 0.0


@dataclass
class GraderResult:
    """Final score and feedback after the episode"""
    score: float
    feedback: str
    breakdown: Dict[str, float]
    time_penalty: float = 0.0
    confidence_bonus: float = 0.0
    improvement_bonus: float = 0.0