---
title: EduNexora OpenEnv
emoji: 📚
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# EduNexora AI — OpenEnv Environment

**Team Nexora AI · OpenEnv Hackathon 2024**

---

## Why We Built This

Every school and college runs on the same broken workflow. Results come out, teachers manually scan spreadsheets to figure out who failed, syllabus tracking happens in WhatsApp groups, and by the time anyone identifies a struggling student, it is already too late.

We built EduNexora because we wanted to see what happens when you put a reinforcement learning environment around that exact problem. Not a toy game. Not a simulation with fake stakes. A system that mirrors what actually happens in an academic institution — result processing, syllabus tracking, and early identification of at-risk students — and trains an agent to do it correctly.

---

## What EduNexora Does

EduNexora is a three-task OpenEnv environment. Each task represents one real workflow from an academic institution. An agent interacts through structured actions, gets rewarded for correct decisions, and builds toward a complete academic intelligence system.

---

## The Three Tasks

### Task 1 — Student Performance Analysis (Easy)

The agent receives student records and classifies each student based on marks.

- Marks 40 and above → Pass
- Marks between 35 and 39 → Fail
- Marks below 35 → Backlog

After classifying all students, the agent generates a ranked leaderboard. Reward is 1.0 for correct classification, 0.5 for accurate ranking, 0.0 for wrong classifications.

### Task 2 — Syllabus Tracking (Medium)

The agent works with a four-unit syllabus and tracks which topics are complete, identifies lagging units, and generates contextual notifications.

Reward scales proportionally with completion progress. A fully completed syllabus returns 1.0. Correct unit prioritization returns 0.5.

### Task 3 — Early Intervention (Hard)

The agent assesses risk and recommends action for each student.

- Marks below 40 → High Risk
- Marks between 40 and 60 → Medium Risk
- Marks above 60 → Low Risk

Correct risk classification returns 1.0. Appropriate intervention assignment returns 0.5. Wrong classification returns 0.0.

---

## Reward System

All rewards are dynamic and in range 0.0 to 1.0. No fixed values. The reward depends on what the agent actually does with the data — wrong decisions get 0.0, correct decisions get full reward, partial progress gets partial reward.

---

## Project Structure

```
edunexora/
├── inference.py          # Main inference script
├── env.py                # OpenEnv environment — reset(), step(), state()
├── models.py             # Pydantic models — Observation, Action, Reward
├── tasks.py              # Task runners for all 3 tasks
├── graders.py            # Reward grading logic
├── app.py                # Flask dashboard
├── openenv.yaml          # OpenEnv configuration
├── Dockerfile            # Docker build
├── requirements.txt      # Dependencies
├── templates/
│   └── index.html        # Dashboard UI
├── static/
│   └── style.css         # Dashboard styling
└── uploads/
    ├── student.csv       # Sample student data for testing
    └── Syallabus.txt     # Sample syllabus data for testing
```

---

## Running Locally

```bash
pip install -r requirements.txt
python inference.py
```

For the full dashboard:

```bash
python app.py
```

Open `http://localhost:7860`

---

## Docker

```bash
docker build -t edunexora .
docker run -p 7860:7860 edunexora
```

The container runs inference logs first, then starts the Flask server on port 7860.

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `API_BASE_URL` | LLM API endpoint | `https://api-inference.huggingface.co/v1/` |
| `MODEL_NAME` | Model identifier | `dummy-model` |
| `HF_TOKEN` | Hugging Face token | empty |

The system runs correctly without these — it uses built-in environment logic and dummy data.

---

## Log Format

```
[START] task=student_analysis env=EduNexoraEnv-v1 model=dummy-model
[STEP] step=1 action=process_all_students reward=0.85
[STEP] step=2 action=process_all_students reward=1.0
[END] success=true steps=101
```

Three complete blocks — one per task.

---

## Dashboard

The Flask frontend shows:

- Student statistics (Pass / Fail / Backlog counts)
- Top 5 ranked students
- Unit-wise syllabus progress
- Risk distribution (High / Medium / Low)
- AI-generated notifications

### Try Real Data

Sample files are included in the `uploads/` folder to test the real data mode.

**Step 1** — Open the Space URL in browser

**Step 2** — Click **"Upload Real Data"** button on the dashboard

**Step 3** — Upload `student.csv` for student analysis (CSV with student names and marks)

**Step 4** — Upload `Syallabus.txt` for syllabus tracking

**Step 5** — Dashboard updates instantly with real AI insights

This demonstrates how the system works with actual institutional data, not just demo data.

---

## OpenEnv Compliance

- `reset()` — initializes state for any of the three tasks
- `step(action)` — processes action, returns observation, reward, done, info
- `state()` — returns current environment state
- All models use Pydantic for type safety
- `openenv.yaml` declares all tasks, action spaces, observation spaces, reward logic

---

## A Note on Design

We could have built a simpler environment. One task, fixed rewards, minimal state.

We chose not to because the whole point of an RL environment is that it should be worth learning from. A system that only teaches an agent to classify three categories is not interesting. A system that teaches classification, progress tracking, and risk-based decision making — and makes the agent responsible for getting all three right — is something worth building.

That is what EduNexora is.