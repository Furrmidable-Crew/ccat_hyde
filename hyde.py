import json
import langchain

from cat.log import log
from cat.mad_hatter.decorators import hook


with open("cat/plugins/ccat_hyde/settings.json", "r") as json_file:
    settings = json.load(json_file)


@hook(priority=1)
def cat_recall_query(user_message, cat):

    # Make a prompt from template
    hypothesis_prompt = langchain.PromptTemplate(
        input_variables=["input"],
        template=settings["hyde_prompt"]
    )

    # Run a LLM chain with the user message as input
    hypothesis_chain = langchain.chains.LLMChain(prompt=hypothesis_prompt, llm=cat._llm)
    answer = hypothesis_chain(user_message)
    log(answer, "INFO")
    
    # Calculate hyde embedding
    cat.working_memory["hyde_embedding"] = cat.embedder(answer["text"])
    return user_message


# Calculates the average between the user's message embedding and the Hyde response embedding
def _calculate_vector_average(config, cat):
    user_embedding = config.embedding
    hyde_embedding = cat.working_memory["hyde_embedding"]
    config.embedding = (user_embedding + hyde_embedding) / 2


@hook(priority=0)
def before_cat_recalls_episodic_memories(config, cat):
    _calculate_vector_average(config, cat)

@hook(priority=0)
def before_cat_recalls_declarative_memories(config, cat):
    _calculate_vector_average(config, cat)

@hook(priority=0)
def before_cat_recalls_procedural_memories(config, cat):
    _calculate_vector_average(config, cat)
