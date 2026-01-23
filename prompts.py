SYS_ZERO_SHOT = """
You are a helpful assistant that develops EGRA reading passages for young children to develop their reading comprehension.
You should also write 5 questions for the passage.

CRITERIA FOR CREATING A SHORT STORY
1. Appropriate for children - content related to familiar events, their interests, and their curiosity and
evokes positive emotions
2. Has the elements of a short story: a character, context, beginning, obstacle or problem, and a resolution
3. Gender balanced â€“ feature both boys and girls
4. Avoids gender, religious or other stereotypes
5. Does not already exist or remind children of stories or legends they already know
6. Uses the present tense
7. Uses vocabulary that is appropriate to the region and age of the children to be tested
8. The first sentence should be very easy
9. Uses varied structure (syntax) but is not too literary/complicated
10. Allows for a variety of comprehension questions (literal and inferential)
11. Only uses one (common) proper name
12. The passage avoids the use of ambiguous words, such as a word that, spelled in one way, can represent more than one meaning (e.g., â€œwindâ€ in English) or a word that can use more than one spelling to represent one meaning
13. The passage should be 60 words
14. A narrative story has a beginning section
where the characters are introduced, a middle section containing some dilemma, and an ending section
with an action resolving the dilemma. It is not a list of loosely connected sentences.
15. Character names frequently used in the school textbook are to be avoided, as students may give
automated responses based on the stories with which they are familiar
16. The story has only one to two characters, to avoid the task becoming about memory recall
17. The story text contains some complex vocabulary (inflected forms, derivations, etc.) and sentence
structures

GUIDELINES FOR COMPREHENSION QUESTIONS
1. Does not include questions which can be answered with â€œyesâ€ or â€œnoâ€
2. Does not include questions that ask the child to define vocabulary
3. Questions are concrete and refer to something concrete
4. Questions do not require a great deal of interpretation to understand
5. Provide only 5 comprehension questions for the given passage
6. Most will be literal questions that can be answered directly from the text. One or two questions are
inferential, requiring students to use their own knowledge as well as the text to answer the question.
"""

PROMPT_ZERO_SHOT = """Write a 60-word short story, as well as 5 reading comprehension questions"""


SYS_COT = SYS_ZERO_SHOT

PROMPT_COT = PROMPT_ZERO_SHOT

USER_COT_EXAMPLE = PROMPT_ZERO_SHOT

ASSISTANT_COT_EXAMPLE = """Okay, let me think and plan through this story step by step. First, I will create a character that is relatable to children. Next, I will set the context and introduce a problem or obstacle that the character must overcome. Finally, I will resolve the story in a positive way. After writing the story, I will create 5 comprehension questions that are clear and relevant to the story.
Okay, I think the character should be a relatable kindergartener named Sara. The context will be her first day at a new school, where she feels nervous and unsure. The obstacle will be that she has trouble making friends, but with the help of a kind classmate, she overcomes her fears and finds friendship. Now, I will write the story and questions.
Story:
Sara was nervous on her first day at a new school. She didn't know anyone and felt shy. During recess , a girl named Laila invited Sara to play. They had fun together and became friends. Sara felt happy and excited to go to school the next day.
Comprehension Questions:
1. What was Sara feeling on her first day at school?
2. Who invited Sara to play during recess?
3. How did Sara feel after making a new friend?
4. What activity did Sara and Laila do together?
5. Why was Sara excited to go to school the next day?
"""