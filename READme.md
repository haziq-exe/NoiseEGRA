Don't have to follow this structure exactly, I just roughly made this:

EGRA_functions.py : Has all the main logic for generation

Model files : Children of EGRA and just call EGRA methods (keeping separate classes for each model incase when it comes time to preference tuning/concept vectors the code changes for each model depending on length etc.)

example main:
```python

from Jais import Jais
from EGRA_functions import EGRA

Jais_model = Jais()

Jais_model.zero_shot(output_file="example.csv", num_stories=1, max_new_tokens=500, do_sample=True, include_sys=True)

# Or you can just do

model = EGRA("model_huggingface_link")
model.zero_shot(output_file="example.csv", num_stories=1, max_new_tokens=500, do_sample=True, include_sys=True)


```
