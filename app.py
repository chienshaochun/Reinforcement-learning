from flask import Flask, render_template, request, jsonify
import numpy as np
import random

app = Flask(__name__)

ACTIONS = ['U', 'D', 'L', 'R']
DIR_TO_ARROW = {'U': '↑', 'D': '↓', 'L': '←', 'R': '→'}
ACTION_TO_DELTA = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}

GAMMA = 0.9
ALPHA = 0.1
EPSILON = 0.1
REWARD = -1
EPISODES = 1000
MAX_STEPS = 200

@app.route("/")
def index():
    return render_template("index.html")

def get_valid_actions(n, state, obstacles):
    actions = []
    i, j = state
    for a, (di, dj) in ACTION_TO_DELTA.items():
        ni, nj = i + di, j + dj
        if 0 <= ni < n and 0 <= nj < n and (ni, nj) not in obstacles:
            actions.append(a)
    return actions or ['U']  # 若無可動作，避免空集

def get_next_state(state, action, n, obstacles):
    i, j = state
    di, dj = ACTION_TO_DELTA[action]
    ni, nj = i + di, j + dj
    if 0 <= ni < n and 0 <= nj < n and (ni, nj) not in obstacles:
        return (ni, nj)
    return (i, j)

def policy_evaluation(n, end, obstacles):
    V = np.zeros((n, n))
    policy = {}
    for i in range(n):
        for j in range(n):
            if (i, j) != end and (i, j) not in obstacles:
                policy[(i, j)] = random.choice(ACTIONS)
    threshold = 1e-4
    while True:
        delta = 0
        new_V = np.copy(V)
        for (i, j), a in policy.items():
            ni, nj = get_next_state((i, j), a, n, obstacles)
            reward = 0 if (i, j) == end else REWARD
            new_V[i, j] = reward + GAMMA * V[ni, nj]
            delta = max(delta, abs(new_V[i, j] - V[i, j]))
        V = new_V
        if delta < threshold:
            break
    return policy, V

def q_learning(n, start, end, obstacles):
    Q = {}
    for i in range(n):
        for j in range(n):
            if (i, j) not in obstacles:
                Q[(i, j)] = {a: 0 for a in ACTIONS}

    for _ in range(EPISODES):
        s = start
        for _ in range(MAX_STEPS):
            if s == end:
                break
            actions = get_valid_actions(n, s, obstacles)
            a = random.choice(actions) if random.random() < EPSILON else max(actions, key=lambda x: Q[s][x])
            s_next = get_next_state(s, a, n, obstacles)
            r = 0 if s_next == end else REWARD
            Q[s][a] += ALPHA * (r + GAMMA * max(Q[s_next].values()) - Q[s][a])
            s = s_next

    policy = {s: max(Q[s], key=Q[s].get) for s in Q if s != end}
    V = np.zeros((n, n))
    for (i, j), a_dict in Q.items():
        V[i, j] = max(a_dict.values())
    return policy, V

def sarsa(n, start, end, obstacles):
    Q = {}
    for i in range(n):
        for j in range(n):
            if (i, j) not in obstacles:
                Q[(i, j)] = {a: 0 for a in ACTIONS}

    for _ in range(EPISODES):
        s = start
        actions = get_valid_actions(n, s, obstacles)
        a = random.choice(actions) if random.random() < EPSILON else max(actions, key=lambda x: Q[s][x])
        for _ in range(MAX_STEPS):
            s_next = get_next_state(s, a, n, obstacles)
            r = 0 if s_next == end else REWARD
            if s_next == end:
                Q[s][a] += ALPHA * (r - Q[s][a])
                break
            next_actions = get_valid_actions(n, s_next, obstacles)
            a_next = random.choice(next_actions) if random.random() < EPSILON else max(next_actions, key=lambda x: Q[s_next][x])
            Q[s][a] += ALPHA * (r + GAMMA * Q[s_next][a_next] - Q[s][a])
            s, a = s_next, a_next

    policy = {s: max(Q[s], key=Q[s].get) for s in Q if s != end}
    V = np.zeros((n, n))
    for (i, j), a_dict in Q.items():
        V[i, j] = max(a_dict.values())
    return policy, V

def policy_iteration(n, end, obstacles):
    V = np.zeros((n, n))
    policy = {}
    for i in range(n):
        for j in range(n):
            if (i, j) != end and (i, j) not in obstacles:
                policy[(i, j)] = random.choice(ACTIONS)

    while True:
        # 評估
        threshold = 1e-4
        while True:
            delta = 0
            new_V = np.copy(V)
            for (i, j), a in policy.items():
                ni, nj = get_next_state((i, j), a, n, obstacles)
                reward = 0 if (i, j) == end else REWARD
                new_V[i, j] = reward + GAMMA * V[ni, nj]
                delta = max(delta, abs(new_V[i, j] - V[i, j]))
            V = new_V
            if delta < threshold:
                break
        # 政策改善
        policy_stable = True
        for (i, j) in policy:
            old_action = policy[(i, j)]
            best_action = max(ACTIONS, key=lambda a: V[get_next_state((i, j), a, n, obstacles)])
            policy[(i, j)] = best_action
            if old_action != best_action:
                policy_stable = False
        if policy_stable:
            break
    return policy, V

@app.route("/submit", methods=["POST"])
def submit():
    data = request.get_json()
    n = data["gridSize"]
    start = (data["start"]["row"], data["start"]["col"])
    end = (data["end"]["row"], data["end"]["col"])
    obstacles = set((o["row"], o["col"]) for o in data["obstacles"])
    algorithm = data.get("algorithm", "policy_evaluation")

    if algorithm == "policy_evaluation":
        policy, V = policy_evaluation(n, end, obstacles)
    elif algorithm == "q_learning":
        policy, V = q_learning(n, start, end, obstacles)
    elif algorithm == "sarsa":
        policy, V = sarsa(n, start, end, obstacles)
    elif algorithm == "policy_iteration":
        policy, V = policy_iteration(n, end, obstacles)
    else:
        return jsonify({"status": "error", "message": f"未知演算法：{algorithm}"}), 400

    cell_info = []
    for i in range(n):
        for j in range(n):
            arrow = ""
            value = None
            if (i, j) not in obstacles and (i, j) != end:
                arrow = DIR_TO_ARROW.get(policy.get((i, j), ''), "")
                value = round(V[i, j], 2)
            elif (i, j) == end:
                value = 0.0
            cell_info.append({
                "row": i,
                "col": j,
                "arrow": arrow,
                "value": value
            })

    return jsonify({"status": "success", "cells": cell_info})
