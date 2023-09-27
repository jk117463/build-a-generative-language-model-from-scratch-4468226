import random
from string import punctuation
from collections import defaultdict


class MarkovChain:
    def __init__(self):
        self.graph = defaultdict(list)

    def _tokenize(self, text):
        return (
            text.translate(str.maketrans("", "", punctuation + "1234567890"))
                .replace("\n", " ")
                .split(" ")
        )

    def train(self, text):
        tokens = self._tokenize(text)
        for i, token in enumerate(tokens):
            if (len(tokens) - 1) == i:
                break
            self.graph[token].append(tokens[i + 1])

    def generate(self, prompt, length=10):
        # get the lask token from the prompt
        current = self._tokenize(prompt)[-1]
        # initialize the output
        output = prompt
        for i in range(length):
            # look up the options in the graph dictionary
            options = self.graph.get(current, [])
            if not options:
                continue
            # use random.choice method to pick a current option
            current = random.choice(options)
            # add the random choice to the output string
            output += f" {current}"
        return output

# Create an instance of the MarkovChain class
mc = MarkovChain()

# Train the Markov chain with some text data
text_data = "This is a simple example of a Markov chain. Markov chains are used in various applications."
mc.train(text_data)

# Generate text using a prompt
generated_text = mc.generate("This is", length=20)
print(generated_text)