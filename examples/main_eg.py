from noiseegra.models.Jais import Jais
from noiseegra.EGRA_functions import EGRA

Jais_model = Jais()

Jais_model.zero_shot(output_file="example.csv", num_stories=1, max_new_tokens=500, do_sample=True, include_sys=True)

model = EGRA(model="inceptionai/Jais-2-70B-Chat")
model.zero_shot(output_file="example2.csv", num_stories=1, max_new_tokens=500, do_sample=True, include_sys=True)
