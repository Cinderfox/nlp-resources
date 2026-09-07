def viterbi(obs, states, start_prob, trans_prob, emission_prob):
    n = len(obs)
    m = len(states)

    V = [[0.0] * m for _ in range(n)]
    path = [[0] * m for _ in range(n)]

    for i in range(m):
        V[0][i] = start_prob[i] * emission_prob[i][obs[0]] if obs[0] in emission_prob[i] else 0.0
        path[0][i] = 0

    for t in range(1, n):
        for j in range(m):
            max_prob, prev_state = max(
                (V[t - 1][i] * trans_prob[i][j] * emission_prob[j][obs[t]] if obs[t] in emission_prob[j] else 0.0, i)
                for i in range(m)
            )
            V[t][j] = max_prob
            path[t][j] = prev_state

    max_prob, last_state = max((V[n - 1][i], i) for i in range(m))

    best_path = [last_state]
    for t in range(n - 1, 0, -1):
        best_path.insert(0, path[t][last_state])
        last_state = path[t][last_state]

    return best_path, max_prob


observed_states = ["The", "cat", "is", "fast"]
hidden_states = ["N", "V", "Adj"]

start_probability = [0.4, 0.3, 0.3]
transition_probability = [
    [0.2, 0.7, 0.1],
    [0.1, 0.6, 0.3],
    [0.3, 0.2, 0.5]
]
emission_probability = [
    {"The": 0.8, "cat": 0.1, "is": 0.1},
    {"The": 0.1, "cat": 0.7, "is": 0.1, "fast": 0.1},
    {"The": 0.2, "cat": 0.1, "is": 0.6, "fast": 0.1}
]

path, probability = viterbi(observed_states, hidden_states, start_probability, transition_probability, emission_probability)

mapped_path = [hidden_states[i] for i in path]


final_sentence = " ".join([f"{observed_states[i]}" for i in range(len(observed_states))])
print("Final Sentence:", final_sentence)
