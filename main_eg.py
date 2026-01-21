from Jais import Jais

Jais_model = Jais()

Jais_model.zero_shot(output_file="example.csv", num_stories=1, max_new_tokens=500, do_sample=True, include_sys=True)