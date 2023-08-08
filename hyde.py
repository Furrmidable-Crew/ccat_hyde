import langchain

from cat.mad_hatter.decorators import hook
from cat.log import log


@hook(priority=1)
def cat_recall_query(user_message, cat):
    # Prompt
    hyde_prompt = """You will be given a sentence.
    If the sentence is a question, convert it to a plausible answer. If the sentence does not contain a question, 
    just repeat the sentence as is without adding anything to it.

    Examples:
    - what furniture there is in my room? --> In my room there is a bed, a wardrobe and a desk with my computer
    - where did you go today --> today I was at school
    - I like ice cream --> I like ice cream
    - how old is Jack --> Jack is 20 years old

    Sentence:
    - {input} --> """  # TODO make this a setting

    # Make a prompt from template
    hypothesis_prompt = langchain.PromptTemplate(
        input_variables=["input"],
        template=hyde_prompt
    )

    # Run a LLM chain with the user message as input
    hypothesis_chain = langchain.chains.LLMChain(prompt=hypothesis_prompt, llm=cat._llm)
    answer = hypothesis_chain(user_message)
    log(answer, "ERROR")
    return answer["text"]
