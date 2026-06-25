"""Model-specific EGRA wrappers.

Each wrapper subclasses :class:`noiseegra.EGRA_functions.EGRA` and pins a
specific Hugging Face model id (and any model-specific setup). Import the one
you need directly, e.g.::

    from noiseegra.models.Jais import Jais
    model = Jais()
"""
