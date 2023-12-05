import json
import langchain

from cat.log import log
from cat.mad_hatter.decorators import hook


with open("cat/plugins/ccat_hyde/settings.json", "r") as json_file:
    settings = json.load(json_file)

# Keys
HYDE_ANSWER       = "hyde_answer"
AVERAGE_EMBEDDING = "average_embedding"


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
    
    # Save HyDE answer in working memory
    cat.working_memory[HYDE_ANSWER] = answer["text"]
    
    print("------------- HYDE -------------")
    print(f"user message: {user_message}")
    print(f"hyde answer: {answer['text']}")
    
    return user_message


# Calculates the average between the user's message embedding and the Hyde response embedding
def _calculate_vector_average(config: dict, cat):
    
    # If average embedding not exists & hyde answer exists ..
    if AVERAGE_EMBEDDING not in cat.working_memory.keys() and HYDE_ANSWER in cat.working_memory.keys():
        
        # Get user message embedding
        user_embedding = config['embedding']
        
        # Calculate hyde embedding from hyde answer
        hyde_answer = cat.working_memory[HYDE_ANSWER]
        hyde_embedding = cat.embedder.embed_query(hyde_answer)

        # Calculate average embedding and stores it into a working memory
        average_embedding = [(x + y)/2 for x, y in zip(user_embedding, hyde_embedding)]
        cat.working_memory[AVERAGE_EMBEDDING] = average_embedding

        #print(f"user_embedding:    {user_embedding}")
        #print(f"hyde_embedding:    {hyde_embedding}")
        #print(f"average_embedding: {average_embedding}")

        # Delete Hyde Answer from working memory
        del cat.working_memory[HYDE_ANSWER]
    
    # If average embedding exists, set the embedding
    if AVERAGE_EMBEDDING in cat.working_memory.keys():
        average_embedding = cat.working_memory[AVERAGE_EMBEDDING]
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
