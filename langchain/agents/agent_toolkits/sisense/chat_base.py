"""Sisense agent."""
from typing import Any, Dict, List, Optional

from langchain.agents import AgentExecutor
from langchain.agents.agent import AgentOutputParser
from langchain.agents.agent_toolkits.sisense.prompt import (
    SISENSE_CHAT_PREFIX,
    SISENSE_CHAT_SUFFIX,
)
from langchain.agents.agent_toolkits.sisense.toolkit import SisenseToolkit
from langchain.agents.conversational_chat.base import ConversationalChatAgent
from langchain.callbacks.base import BaseCallbackManager
from langchain.chat_models.base import BaseChatModel
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_memory import BaseChatMemory
from langchain.utilities.sisense import SisenseDataset


def create_sisense_chat_agent(
    llm: BaseChatModel,
    toolkit: Optional[SisenseToolkit],
    sisense: Optional[SisenseDataset] = None,
    callback_manager: Optional[BaseCallbackManager] = None,
    output_parser: Optional[AgentOutputParser] = None,
    prefix: str = SISENSE_CHAT_PREFIX,
    suffix: str = SISENSE_CHAT_SUFFIX,
    examples: Optional[str] = None,
    input_variables: Optional[List[str]] = None,
    memory: Optional[BaseChatMemory] = None,
    top_k: int = 10,
    verbose: bool = False,
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Dict[str, Any],
) -> AgentExecutor:
    """Construct a sisense agent from an Chat LLM and tools.

    If you supply only a toolkit and no sisense dataset, the same LLM is used for both.
    """
    if toolkit is None:
        if sisense is None:
            raise ValueError("Must provide either a toolkit or sisense dataset")
        toolkit = SisenseToolkit(sisense=sisense, llm=llm, examples=examples)
    tools = toolkit.get_tools()
    agent = ConversationalChatAgent.from_llm_and_tools(
        llm=llm,
        tools=tools,
        system_message=prefix.format(top_k=top_k),
        human_message=suffix,
        input_variables=input_variables,
        callback_manager=callback_manager,
        output_parser=output_parser,
        verbose=verbose,
        **kwargs,
    )
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        callback_manager=callback_manager,
        memory=memory
        or ConversationBufferMemory(memory_key="chat_history", return_messages=True),
        verbose=verbose,
        **(agent_executor_kwargs or {}),
    )
