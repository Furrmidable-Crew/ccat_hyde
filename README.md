# Ccat Hyde

[![Awesome plugin](https://custom-icon-badges.demolab.com/static/v1?label=&message=Awesome+plugin&color=000000&style=for-the-badge&logo=cheshire_cat_ai)](https://)  

# Hypothetical Document Embedding Plugin

This plugin enables the [Hypothetical Document Embedding](https://cheshire-cat-ai.github.io/docs/conceptual/prompts/hyde/) (HyDE) technique.

HyDE consists in asking the Language Model to simulate an answer (*hypothetical* answer) to a question.
Such answer is [embedded](https://cheshire-cat-ai.github.io/docs/conceptual/llm/#embedding-model)
and used
to recall relevant context from the [vector memories](https://cheshire-cat-ai.github.io/docs/conceptual/memory/vector_memory/).

HyDE is beneficial in Question-Answering tasks.
The underlying idea is that using the embedded *hypothetical* answer in a vector similarity search would lead to better results
rather than using the embedded question itself.

> **Warning**
> When this plugin is enabled, the Cat makes and additional call to the language model API.
