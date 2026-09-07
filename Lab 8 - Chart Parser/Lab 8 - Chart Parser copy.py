import nltk


class Rule:
    def __init__(self, left, right):
        self.left = left
        self.right = right


class ChartParser:
    def __init__(self, grammar):
        self.grammar = grammar
        self.chart = []

    def predict(self, state):
        for rule in self.grammar:
            if state.next() == rule.right[0]:
                self.chart.append(state.advance(rule))

    def scan(self, state, token):
        if state.next() == token:
            self.chart.append(state.advance())

    def complete(self, state):
        for st in self.chart:
            if not st.is_complete() and st.next() == state.rule.left:
                self.chart.append(st.advance())

    def parse(self, tokens):
        start_state = State(self.grammar[0], 0, 0, 0)
        self.chart.append(start_state)

        for i, token in enumerate(tokens):
            for state in self.chart[:]:
                if not state.is_complete():
                    if state.next_is_nonterminal():
                        self.predict(state)
                    else:
                        self.scan(state, token)
            for state in self.chart[:]:
                if state.is_complete():
                    self.complete(state)

        return self.chart

    def print_chart(self):
        print("Chart:")
        for i, state in enumerate(self.chart):
            print(f"{i}: {state}")


class State:
    def __init__(self, rule, dot, start, end):
        self.rule = rule
        self.dot = dot
        self.start = start
        self.end = end

    def next(self):
        return self.rule.right[self.dot]

    def next_is_nonterminal(self):
        return isinstance(self.next(), str)

    def is_complete(self):
        return self.dot == len(self.rule.right)

    def advance(self, rule=None):
        if rule is None:
            rule = self.rule
        return State(rule, self.dot + 1, self.start, self.end + 1)

    def __repr__(self):
        return f"{self.rule.left} -> {' '.join(self.rule.right[:self.dot])} . {' '.join(self.rule.right[self.dot:])} [{self.start}:{self.end}]"


grammar = [
    Rule("S", ["NP", "VP"]),
    Rule("NP", ["Det", "N"]),
    Rule("VP", ["V", "NP"]),
    Rule("Det", ["the"]),
    Rule("N", ["cat"]),
    Rule("V", ["chased"])
]

tokens = ["the", "cat", "chased"]

chart_parser = ChartParser(grammar)
chart = chart_parser.parse(tokens)
chart_parser.print_chart()
