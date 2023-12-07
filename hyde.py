from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from cat.mad_hatter.decorators import plugin
from pydantic import BaseModel, Field
from cat.log import log
from cat.mad_hatter.decorators import hook


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


# Keys
HYDE_ANSWER       = "hyde_answer"
AVERAGE_EMBEDDING = "average_embedding"


@hook(priority=1)
def cat_recall_query(user_message, cat):

    # Acquire settings
    settings = cat.mad_hatter.get_plugin().load_settings() 
    log.debug(f" --------- ACQUIRE SETTINGS ---------")
    log.debug(f"settings: {settings}")

    # Make a prompt from template
    hypothesis_prompt = PromptTemplate(
        input_variables=["input"],
        template=settings["hyde_prompt"]
    )

    # Run a LLM chain with the user message as input
    hypothesis_chain = LLMChain(prompt=hypothesis_prompt, llm=cat._llm)
    answer = hypothesis_chain(user_message)
    
    # Save HyDE answer in working memory
    cat.working_memory[HYDE_ANSWER] = answer["text"]
    
    log.warning("------------- HYDE -------------")
    log.warning(f"user message: {user_message}")
    log.warning(f"hyde answer: {answer['text']}")
    
    return user_message


# Calculates the average between the user's message embedding and the Hyde response embedding
def _calculate_vector_average(config: dict, cat):
    
    # If hyde answer exists, calculate and set average embedding
    if HYDE_ANSWER in cat.working_memory.keys():
        
       # Get user message embedding
        user_embedding = config['embedding']
        
        # Calculate hyde embedding from hyde answer
        hyde_answer = cat.working_memory[HYDE_ANSWER]
        hyde_embedding = cat.embedder.embed_query(hyde_answer)

        # Calculate average embedding and stores it into a working memory
        average_embedding = [(x + y)/2 for x, y in zip(user_embedding, hyde_embedding)]
        cat.working_memory[AVERAGE_EMBEDDING] = average_embedding

        log.debug(f" --------- CALCULATE AVERAGE ---------")
        log.debug(f"hyde answer:       {hyde_answer}")
        log.debug(f"user_embedding:    {user_embedding}")
        log.debug(f"hyde_embedding:    {hyde_embedding}")
        log.debug(f"average_embedding: {average_embedding}")

        # Delete Hyde Answer from working memory
        del cat.working_memory[HYDE_ANSWER]

    # If average embedding exists, set the embedding
    if AVERAGE_EMBEDDING in cat.working_memory.keys():
        average_embedding = cat.working_memory[AVERAGE_EMBEDDING]
        config['embedding'] = average_embedding
        
        log.debug(f" --------- SET EMBEDDING ---------")
        log.debug(f"average_embedding: {average_embedding}")
        

@hook(priority=1)
def before_cat_recalls_episodic_memories(config: dict, cat):
    _calculate_vector_average(config, cat)

@hook(priority=1)
def before_cat_recalls_declarative_memories(config: dict, cat):
    _calculate_vector_average(config, cat)

@hook(priority=1)
def before_cat_recalls_procedural_memories(config: dict, cat):
    _calculate_vector_average(config, cat)
