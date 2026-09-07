class State(object):
    def __init__(self, label, rules, dot_idx, start_idx, end_idx, idx, made_from, producer):
        self.label = label
        self.rules = rules
        self.dot_idx = dot_idx
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.idx = idx
        self.made_from = made_from
        self.producer = producer

    def next(self):
        return self.rules[self.dot_idx]

    def complete(self):
        return len(self.rules) == self.dot_idx

    def __eq__(self, other):
        return (self.label == other.label and
                self.rules == other.rules and
                self.dot_idx == other.dot_idx and
                self.start_idx == other.start_idx and
                self.end_idx == other.end_idx)

    def __str__(self):
        rule_string = ''
        for i, rule in enumerate(self.rules):
            if i == self.dot_idx:
                rule_string += '•\t\t '
            rule_string += rule + ' '
        if self.dot_idx == len(self.rules):
            rule_string += '•\t\t'
        return 'S%d %s -> %s [%d, %d] %s %s' % (self.idx, self.label, rule_string, self.start_idx, 
                                                self.end_idx, self.made_from, self.producer)

class Earley:
    def __init__(self, words, grammar_rules, pos_name):
        self.chart = [[] for _ in range(len(words) + 1)]
        self.current_id = 0
        self.words = words
        self.grammar_rules = grammar_rules
        self.pos_name = pos_name

    def generate_unique_id(self):
        self.current_id += 1
        return self.current_id - 1

    def is_terminal_symbol(self, tag):
        return tag in self.pos_name

    def is_state_complete(self, state):
        return len(state.rules) == state.dot_idx

    def add_state_to_chart(self, state, chart_entry):
        if state not in self.chart[chart_entry]:
            self.chart[chart_entry].append(state)
        else:
            self.current_id -= 1

    def perform_prediction(self, state):
        for production in self.grammar_rules[state.next()]:
            self.add_state_to_chart(State(state.next(), production, 0, state.end_idx, state.end_idx, self.generate_unique_id(), [], 'perform_prediction'), state.end_idx)

    def perform_scanning(self, state):
        if self.words[state.end_idx] in self.grammar_rules[state.next()]:
            self.add_state_to_chart(State(state.next(), [self.words[state.end_idx]], 1, state.end_idx, state.end_idx + 1, self.generate_unique_id(), [], 'perform_scanning'), state.end_idx + 1)

    def perform_completion(self, state):
        for s in self.chart[state.start_idx]:
            if not s.complete() and s.next() == state.label and s.end_idx == state.start_idx and s.label != 'gamma':
                self.add_state_to_chart(State(s.label, s.rules, s.dot_idx + 1, s.start_idx, state.end_idx, self.generate_unique_id(), s.made_from + [state.idx], 'perform_completion'), state.end_idx)

    def start_parsing(self):
        self.add_state_to_chart(State('gamma', ['S'], 0, 0, 0, self.generate_unique_id(), [], 'dummy start state'), 0)
        
        for i in range(len(self.words) + 1):
            for state in self.chart[i]:
                if not state.complete() and not self.is_terminal_symbol(state.next()):
                    self.perform_prediction(state)
                elif i != len(self.words) and not state.complete() and self.is_terminal_symbol(state.next()):
                    self.perform_scanning(state)
                else:
                    self.perform_completion(state)

    def __str__(self):
        res = ''
        
        for i, chart in enumerate(self.chart):
            res += '\nChart[%d]\n' % i
            for state in chart:
                res += str(state) + '\n'

        return res



grammar_rules = {
    'S':           [['NP', 'VP'], ['Aux', 'NP', 'VP'], ['VP']],
    'NP':          [['Det', 'Nominal'], ['Proper-Noun']],
    'Nominal':     [['Noun'], ['Noun', 'Nominal']],
    'VP':          [['Verb'], ['Verb', 'NP']],
    'Det':         ['the', 'a'],
    'Noun':        ['cat', 'dog', 'bird'],
    'Verb':        ['chased', 'ate', 'flew'],
    'Aux':         ['does'],
    'Proper-Noun': ['Tom', 'Jerry']
}

pos_name = ['Det', 'Noun', 'Verb', 'Aux', 'Prep', 'Proper-Noun']

earleyAlgo = Earley(['Tom', 'chased', 'the', 'cat'], grammar_rules, pos_name)
earleyAlgo.start_parsing()
print (earleyAlgo)