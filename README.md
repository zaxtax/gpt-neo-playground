# gpt-neo-playground
A playground to explores large language models

To try this out first install `transformers`

````bash
pip install torch transformers
````

Then use inference to try out some prompts:

````python
from demo import inference
inference("# Returns a list of n primes")
````

With the Facebook model, infill is supported

````python
prompt = """
def count_words(filename: str) -> <|mark:0|i>
    \"""Count the number of occurrences of each word in the file.\"""
    with open(filename, 'r') as f:
        word_counts = {}
        for line in f:
            for word in line.split():
                if word in word_counts:
                    word_counts[word]  = 1
                else:
                    word_counts[word] = 1
    return word_counts
"""
inference(prompt)
````
