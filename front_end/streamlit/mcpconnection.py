import streamlit as st
import os

import front_end.mcpServer.clienttest as mcpServer

async def run_mcp():
    config = mcpServer.Configuration()
    server_config = config.load_config("front_end/mcpServer/servers_config.json")
    servers = [
        mcpServer.Server(name, srv_config)
        for name, srv_config in server_config["mcpServers"].items()
    ]
    llm_client = mcpServer.LLMClient(config.llm_api_key)
    chat_session = mcpServer.ChatSession(servers, llm_client)
    await chat_session.start()
    
def mcp_connector():
    import asyncio
    asyncio.run(run_mcp())
    

    