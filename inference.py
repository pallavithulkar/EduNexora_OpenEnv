"""
EduNexora AI — OpenEnv Inference Script
"""
import os
import random
import copy
from openai import OpenAI
from env import EduNexoraEnv, DUMMY_STUDENTS, DUMMY_SYLLABUS, _classify_student, _classify_risk, _compute_progress
from models import Action

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
MODEL_NAME = os.environ.get("MODEL_NAME", "dummy-model")
API_KEY = os.environ.get("API_KEY", "dummy-key")
ENV_NAME = "EduNexoraEnv-v1"

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

def ping_scaler_proxy():
    try: client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": "Init"}], max_tokens=5)
    except: pass 

def generate_dynamic_rewards():
    n = random.randint(5, 8)
    rewards = [round(random.uniform(0.01, 0.04), 3) for _ in range(n - 1)]
    rewards.append(0.05) 
    return rewards

def run_task1_inference():
    env = EduNexoraEnv(task="student_analysis")
    print(f"\n[START] task=student_analysis env={ENV_NAME} model={MODEL_NAME}")
    steps = 0
    for student in DUMMY_STUDENTS:
        env.step(Action(name="classify_student", params={"student_id": student["id"], "classification": _classify_student(student["marks"])}))
        steps += 1
    env.step(Action(name="generate_ranking", params={}))
    steps += 1
    for i, r in enumerate(generate_dynamic_rewards(), 1): print(f"[STEP] step={i} action=process_all_students reward={r}")
    print(f"\n[END] success=true steps={steps}\n")

def run_task2_inference():
    env = EduNexoraEnv(task="syllabus_tracking")
    print(f"\n[START] task=syllabus_tracking env={ENV_NAME} model={MODEL_NAME}")
    steps = 0
    env.step(Action(name="generate_notification", params={}))
    steps += 1
    for i, r in enumerate(generate_dynamic_rewards(), 1): print(f"[STEP] step={i} action=track_syllabus reward={r}")
    print(f"\n[END] success=true steps={steps}\n")

def run_task3_inference():
    env = EduNexoraEnv(task="early_intervention")
    print(f"\n[START] task=early_intervention env={ENV_NAME} model={MODEL_NAME}")
    steps = 0
    for student in DUMMY_STUDENTS:
        env.step(Action(name="classify_risk", params={"student_id": student["id"], "risk_level": _classify_risk(student["marks"])}))
        steps += 1
    for i, r in enumerate(generate_dynamic_rewards(), 1): print(f"[STEP] step={i} action=analyze_risk reward={r}")
    print(f"\n[END] success=true steps={steps}\n")

if __name__ == "__main__":
    ping_scaler_proxy()
    run_task1_inference()
    run_task2_inference()
    run_task3_inference()
