SYS_ZERO_SHOT = """
أنتَ مساعدٌ مُفيدٌ تُعدّ نصوص قراءةٍ مُخصصةٍ للأطفال الصغار لتنمية مهاراتهم في فهم المقروء.

يُرجى أيضًا كتابة خمسة أسئلةٍ لكل نص.
"""

PROMPT_ZERO_SHOT = """
اكتب قصه.
* يجب أن تكون القصة سردية مستوحاة من مواد قراءة الأطفال، وتتضمن:
  * مقدمة تُعرّف بالشخصيات
  * جزءًا وسطيًا يتضمن معضلة ما
  * جزءًا ختاميًا يتضمن حدثًا لحل المعضلة
* يجب ألا تتجاوز القصة 60 كلمة.
* يجب أن تدور القصة حول شخصية أو شخصيتين، بأسماء شائعة في اللغة العربية وسياق الطفل، ولكنها غير شائعة الاستخدام في الكتب المدرسية.
* مناسب للأطفال – محتوى مرتبط بأحداث مألوفة واهتماماتهم وفضولهم، ويثير مشاعر إيجابية.
* يحتوي على عناصر القصة القصيرة: شخصية، سياق، بداية، عقبة أو مشكلة، وحل.
* متوازن بين الجنسين – يضم كلاً من الأولاد والبنات.
* يتجنب الصور النمطية المتعلقة بالجنس أو الدين أو غيرها.
* لا يوجد نص موجود مسبقًا ولا يذكر الأطفال بقصص أو أساطير يعرفونها.
* يستخدم زمن المضارع.
* يستخدم مفردات مناسبة للمنطقة والفئة العمرية للأطفال الذين سيتم اختبارهم.
* يجب أن تكون الجملة الأولى سهلة للغاية.
* يستخدم بنية متنوعة ولكنها ليست أدبية أو معقدة.
* يسمح بطرح أسئلة فهم متنوعة (حرفية واستنتاجية).
* يستخدم اسمًا علمًا واحدًا (شائعًا) فقط.
* تتجنب القصة استخدام الكلمات المبهمة، مثل كلمة يمكن أن تدل على أكثر من معنى عند كتابتها بطريقة معينة، أو كلمة يمكن أن تُستخدم أكثر من تهجئة لتمثيل معنى واحد.
* ليس قائمة من الجمل المترابطة بشكل ضعيف.
* يجب تجنب استخدام أسماء الشخصيات الشائعة في الكتب المدرسية، لأن الطلاب قد يقدمون إجابات تلقائية بناءً على القصص التي يعرفونها.
* تحتوي القصة على شخصية أو شخصيتين فقط، لتجنب تحول المهمة إلى اختبار للذاكرة.
* يحتوي نص القصة على بعض المفردات المعقدة وتراكيب الجمل.
"""


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