"""Wrapper around a sisense endpoint."""
from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Union

import aiohttp
import requests
from aiohttp import ServerTimeoutError
from pydantic import BaseModel, Field, root_validator, validator
from requests.exceptions import Timeout

_LOGGER = logging.getLogger(__name__)

BASE_URL = os.getenv("SISENSE_BASE_URL", "https://api.SISENSE.com/v2.0/myorg")

if TYPE_CHECKING:
    from azure.core.credentials import TokenCredential


class sisenseNLQ(BaseModel):
    """Create an NLQ.

    you are a data analyst, your job is to recieve a database schema and a question and identify the columns that are relevant to answer the question. 
    step 1 - understand the question, 
    step 2 - identify columns that are relevant to the question, 
    step 3 - identify relevent aggregation or function to answer the question, 
    step 4 - identify relevent columns and their values as filters to answer the question. 
    """


    @property
    def request_url(self) -> str:
        """Get the request url."""
        return f"{BASE_URL}/nlq/query"

    @property
    def headers(self) -> Dict[str, str]:
        """Get the token."""
        if self.token:
            return {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.token,
            }
        from azure.core.exceptions import (
            ClientAuthenticationError,  # pylint: disable=import-outside-toplevel
        )

        if self.credential:
            try:
                token = self.credential.get_token(
                    "https://analysis.windows.net/SISENSE/api/.default"
                ).token
                return {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + token,
                }
            except Exception as exc:  # pylint: disable=broad-exception-caught
                raise ClientAuthenticationError(
                    "Could not get a token from the supplied credentials."
                ) from exc
        raise ClientAuthenticationError("No credential or token supplied.")

    def get_table_names(self) -> Iterable[str]:
        """Get names of tables available."""
        return self.table_names

    def get_schemas(self) -> str:
        """Get the available schema's."""
        if self.schemas:
            return ", ".join([f"{key}: {value}" for key, value in self.schemas.items()])
        return "No known schema's yet. Use the schema_SISENSE tool first."

    @property
    def table_info(self) -> str:
        """Information about all tables in the database."""
        return self.get_table_info()

    def _get_tables_to_query(
        self, table_names: Optional[Union[List[str], str]] = None
    ) -> Optional[List[str]]:
        """Get the tables names that need to be queried, after checking they exist."""
        if table_names is not None:
            if (
                isinstance(table_names, list)
                and len(table_names) > 0
                and table_names[0] != ""
            ):
                fixed_tables = [fix_table_name(table) for table in table_names]
                non_existing_tables = [
                    table for table in fixed_tables if table not in self.table_names
                ]
                if non_existing_tables:
                    _LOGGER.warning(
                        "Table(s) %s not found in dataset.",
                        ", ".join(non_existing_tables),
                    )
                tables = [
                    table for table in fixed_tables if table not in non_existing_tables
                ]
                return tables if tables else None
            if isinstance(table_names, str) and table_names != "":
                if table_names not in self.table_names:
                    _LOGGER.warning("Table %s not found in dataset.", table_names)
                    return None
                return [fix_table_name(table_names)]
        return self.table_names

    def _get_tables_todo(self, tables_todo: List[str]) -> List[str]:
        """Get the tables that still need to be queried."""
        return [table for table in tables_todo if table not in self.schemas]

    def _get_schema_for_tables(self, table_names: List[str]) -> str:
        """Create a string of the table schemas for the supplied tables."""
        schemas = [
            schema for table, schema in self.schemas.items() if table in table_names
        ]
        return ", ".join(schemas)

    def get_table_info(
        self, table_names: Optional[Union[List[str], str]] = None
    ) -> str:
        """Get information about specified tables."""
        tables_requested = self._get_tables_to_query(table_names)
        if tables_requested is None:
            return "No (valid) tables requested."
        tables_todo = self._get_tables_todo(tables_requested)
        for table in tables_todo:
            self._get_schema(table)
        return self._get_schema_for_tables(tables_requested)

    async def aget_table_info(
        self, table_names: Optional[Union[List[str], str]] = None
    ) -> str:
        """Get information about specified tables."""
        tables_requested = self._get_tables_to_query(table_names)
        if tables_requested is None:
            return "No (valid) tables requested."
        tables_todo = self._get_tables_todo(tables_requested)
        await asyncio.gather(*[self._aget_schema(table) for table in tables_todo])
        return self._get_schema_for_tables(tables_requested)

    def _get_schema(self, table: str) -> None:
        """Get the schema for a table."""
        try:
            result = self.run(
                f"EVALUATE TOPN({self.sample_rows_in_table_info}, {table})"
            )
            self.schemas[table] = json_to_md(result["results"][0]["tables"][0]["rows"])
        except Timeout:
            _LOGGER.warning("Timeout while getting table info for %s", table)
            self.schemas[table] = "unknown"
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _LOGGER.warning("Error while getting table info for %s: %s", table, exc)
            self.schemas[table] = "unknown"

    async def _aget_schema(self, table: str) -> None:
        """Get the schema for a table."""
        try:
            result = await self.arun(
                f"EVALUATE TOPN({self.sample_rows_in_table_info}, {table})"
            )
            self.schemas[table] = json_to_md(result["results"][0]["tables"][0]["rows"])
        except ServerTimeoutError:
            _LOGGER.warning("Timeout while getting table info for %s", table)
            self.schemas[table] = "unknown"
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _LOGGER.warning("Error while getting table info for %s: %s", table, exc)
            self.schemas[table] = "unknown"

    def _create_json_content(self, command: str) -> dict[str, Any]:
        """Create the json content for the request."""
        return {
            "queries": [{"query": rf"{command}"}],
            "impersonatedUserName": self.impersonated_user_name,
            "serializerSettings": {"includeNulls": True},
        }

    def run(self, command: str) -> Any:
        """Execute a DAX command and return a json representing the results."""
        _LOGGER.debug("Running command: %s", command)
        result = requests.post(
            self.request_url,
            json=self._create_json_content(command),
            headers=self.headers,
            timeout=10,
        )
        return result.json()

    async def arun(self, command: str) -> Any:
        """Execute a DAX command and return the result asynchronously."""
        _LOGGER.debug("Running command: %s", command)
        if self.aiosession:
            async with self.aiosession.post(
                self.request_url,
                headers=self.headers,
                json=self._create_json_content(command),
                timeout=10,
            ) as response:
                response_json = await response.json()
                return response_json
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.request_url,
                headers=self.headers,
                json=self._create_json_content(command),
                timeout=10,
            ) as response:
                response_json = await response.json()
                return response_json


def json_to_md(
    json_contents: List[Dict[str, Union[str, int, float]]],
    table_name: Optional[str] = None,
) -> str:
    """Converts a JSON object to a markdown table."""
    output_md = ""
    headers = json_contents[0].keys()
    for header in headers:
        header.replace("[", ".").replace("]", "")
        if table_name:
            header.replace(f"{table_name}.", "")
        output_md += f"| {header} "
    output_md += "|\n"
    for row in json_contents:
        for value in row.values():
            output_md += f"| {value} "
        output_md += "|\n"
    return output_md


def fix_table_name(table: str) -> str:
    """Add single quotes around table names that contain spaces."""
    if " " in table and not table.startswith("'") and not table.endswith("'"):
        return f"'{table}'"
    return table
