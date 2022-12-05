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
