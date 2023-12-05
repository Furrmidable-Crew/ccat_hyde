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
    cat.working_memory["hyde_embedding"] = cat.embedder.embed_query(answer["text"])
    return user_message


# Calculates the average between the user's message embedding and the Hyde response embedding
def _calculate_vector_average(config: dict, cat):
    user_embedding = config['embedding']
    hyde_embedding = cat.working_memory["hyde_embedding"]
    average_embedding = [(x + y)/2 for x, y in zip(user_embedding, hyde_embedding)]
    config['embedding'] = average_embedding


@hook(priority=1)
def before_cat_recalls_episodic_memories(config: dict, cat):
    _calculate_vector_average(config, cat)

@hook(priority=1)
def before_cat_recalls_declarative_memories(config: dict, cat):
    _calculate_vector_average(config, cat)

@hook(priority=1)
def before_cat_recalls_procedural_memories(config: dict, cat):
    _calculate_vector_average(config, cat)
