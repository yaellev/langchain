"""Toolkit for interacting with a Power BI dataset."""
from typing import List, Optional

from pydantic import Field

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from langchain.tools.sisense.prompt import QUESTION_TO_QUERY
from langchain.tools.sisense.tool import (
    InfoSisenseTool,
    ListSisenseTool,
    QuerySisenseTool,
)
from langchain.utilities.sisense import SisenseDataset


class SisenseToolkit(BaseToolkit):
    """Toolkit for interacting with Sisense dataset."""

    sisense: SisenseDataset = Field(exclude=True)
    llm: BaseLanguageModel = Field(exclude=True)
    examples: Optional[str] = None
    max_iterations: int = 5
    callback_manager: Optional[BaseCallbackManager] = None

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        if self.callback_manager:
            chain = LLMChain(
                llm=self.llm,
                callback_manager=self.callback_manager,
                prompt=PromptTemplate(
                    template=QUESTION_TO_QUERY,
                    input_variables=["tool_input", "tables", "schemas", "examples"],
                ),
            )
        else:
            chain = LLMChain(
                llm=self.llm,
                prompt=PromptTemplate(
                    template=QUESTION_TO_QUERY,
                    input_variables=["tool_input", "tables", "schemas", "examples"],
                ),
            )
        return [
            QuerySisenseTool(
                llm_chain=chain,
                sisense=self.sisense,
                examples=self.examples,
                max_iterations=self.max_iterations,
            ),
            InfoSisenseTool(sisense=self.sisense),
            ListSisenseTool(sisense=self.sisense),
        ]
