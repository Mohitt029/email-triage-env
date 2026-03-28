---
title: Email Triage Environment
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
sdk_version: "1.0.0"
app_file: server/app.py
pinned: false
---

# Email Triage Environment

A production-ready OpenEnv environment for training AI agents to perform real-world email triage tasks.

## Overview

This environment simulates an email assistant who must read, categorize, prioritize, and respond to emails. It's designed for training RL agents to handle real-world communication workflows that humans perform daily.

### Real-World Utility

Email management consumes 28% of the average workweek. Companies spend millions on email handling. This environment enables:
- Training AI assistants that can triage email efficiently
- Evaluating agent performance on realistic tasks
- Developing systems that reduce human email management time

## Tasks

### Easy Task
- **Scenario**: Clear urgent production issue
- **Email**: "URGENT: Server Down - Production Impact"
- **Expected**: Category=urgent, Priority=1, Action=reply
- **Challenge Level**: Basic pattern recognition

### Medium Task  
- **Scenario**: Ambiguous client follow-up
- **Email**: "Question about our recent meeting"
- **Expected**: Category=important, Priority=3, Action=reply
- **Challenge Level**: Requires judgment and context

### Hard Task
- **Scenario**: Complex multi-email thread
- **Context**: Budget review thread with escalating urgency
- **Expected**: Category=urgent, Priority=1, Action=reply  
- **Challenge Level**: Requires thread comprehension and urgency assessment

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| category | string | urgent, important, normal, spam |
| priority | integer | 1-5 (1=highest) |
| action | string | reply, archive, delegate, flag, delete |
| response_text | string | Required if action="reply" |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| current_email | object | Full email content with metadata |
| task_description | string | Task objective |
| task_difficulty | string | easy/medium/hard |
| previous_decisions | list | History of actions |
| done | boolean | Episode completion status |
| reward | float | Immediate reward |

## Reward Structure

The reward function provides partial credit throughout the episode:

| Component | Weight | Description |
|-----------|--------|-------------|
| Category | 35% | Correct email classification |
| Priority | 30% | Appropriate urgency level |
| Action | 25% | Correct action selection |
| Response | 10% | Response quality (if applicable) |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start new episode |
| `/step` | POST | Take action |
| `/state` | GET | Get current state |
| `/tasks` | GET | List tasks |
| `/grader` | POST | Get episode score |
| `/health` | GET | Health check |

## Baseline Results

Using GPT-3.5-turbo as baseline agent:

| Task | Score |
|------|-------|
| Easy | 0.91 |
| Medium | 0.85 |
| Hard | 0.70 |
| **Average** | **0.82** |

## Usage

### Connect to the Environment

```python
from client import EmailTriageEnv
from models import EmailTriageAction

# Connect to environment
env = EmailTriageEnv(base_url="https://MohitS029-email-triage-env.hf.space")

# Start episode
result = env.reset()
print(f"Email: {result.observation.current_email.subject}")

# Make decision
action = EmailTriageAction(
    category="urgent",
    priority=1,
    action="reply",
    response_text="Investigating immediately"
)

# Execute
result = env.step(action)
print(f"Reward: {result.reward}")

# Get final score
score = env.get_grader_score()
print(f"Score: {score}")

env.close()