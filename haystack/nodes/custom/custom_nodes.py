from haystack.nodes.base import BaseComponent
from haystack.schema import Document
from time import sleep
from typing import Any, List, Optional, Union


class TableGate(BaseComponent):
    outgoing_edges = 2

    def run(self, query: str, open: Optional[bool] = True):
        """This allows us to gate each Weaviate table individually in the pipeline"""
        # only when the gate is set to 'open' will be sending the query through 'output_1' (which is the one used downstream)
        if open:
            return {"output_1": query, "_debug": {"open": open}}, "output_1"
        else:
            # when not allowed through, then we send it to a black hole output, output_2, which is unused
            return {"output_2": query, "_debug": {"open": open}}, "output_2"

    def run_batch(self, queries: List[str], open: Optional[bool] = True):
        # not implemented
        return {}, "split"


# if we want to use the same PromptModel in multiple pipelines with different PromptTemplates,
# then we need to have our own custom module for building the prompt and supply the already constructed prompt
# to the LLM model (Prompt Node)
class PromptBuilder(BaseComponent):
    outgoing_edges = 1

    def run(self, query: str, documents: List[Document], prompt_template: Optional[str] = "$query"):
        """Take the merged document, the query and the prompt template and create the Prompt out of those
        And return the query as the 'prompt', as our liberal Prompt Template in the Pipelin takes the $query only,
        not the documents, as we are the ones who are constructing the template, not Haystack

        We do this, because we want to build the Prompt at runtime, so we can have the same large LLM model
        serving multiple types of prompts.
        """
        # this node should be after a document merger, so it should be only getting 1 document
        if not len(documents) <= 1:
            raise ValueError(
                "The PromptBuilder custom node has received more than 1 documents - something must be wrong upstream!"
            )
        doc = documents[0]

        # construct the prompt
        if prompt_template:
            prompt = prompt_template.replace("$query", query).replace("$context", doc.content)
        else:
            prompt = query

        # we will be returning the prompt as the 'query', as that is what our PromptNode is expecting (and the documents can be empty)
        return {
            "documents": [Document(content="")],
            "query": prompt,
            "_debug": {"prompt_template": prompt_template},
        }, "output_1"

    def run_batch(
        self,
        queries: List[str],
        documents: Union[List[Document], List[List[Document]]],
        prompt_template: Optional[str] = "$query",
    ):
        # not implemented
        return {}, "split"
