from cat.mad_hatter.decorators import plugin
from pydantic import BaseModel, Field


class MySettings(BaseModel):
    hyde_prompt: str = Field(
                title="HyDe prompt",
                default="""You will be given a sentence.
    If the sentence is a question, convert it to a plausible answer. If the sentence does not contain a question, 
    just repeat the sentence as is without adding anything to it.

    Examples:
    - what furniture there is in my room? --> In my room there is a bed, a wardrobe and a desk with my computer
    - where did you go today --> today I was at school
    - I like ice cream --> I like ice cream
    - how old is Jack --> Jack is 20 years old

    Sentence:
    - {input} -->""",
                extra={"type": "TextArea"}
        )


@plugin
def settings_schema():
    return MySettings.schema()
