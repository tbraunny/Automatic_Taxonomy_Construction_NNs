import streamlit as st
import os

import src.mcp.clienttest as mcpServer
import asyncio

def run_async_sync(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # No loop running
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

async def run_mcp():
    config = mcpServer.Configuration()
    server_config = config.load_config("src/mcp/servers_config.json")
    servers = [
        mcpServer.Server(name, srv_config)
        for name, srv_config in server_config["mcpServers"].items()
    ]
    llm_client = mcpServer.LLMClient(config.llm_api_key)
    chat_session = mcpServer.ChatSession(servers, llm_client)
    #await chat_session.start()
    await chat_session.initialize()
    return chat_session



async def mcp_connector():
    mcp_server = await run_mcp()
   # asyncio.run(mcp_server.initialize())
    #mcp_server.query_server()
        # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.title("Chat with AI on your Ontology")
    st.write("Ask about neural network classes or instances from the ontology.")

    # Display previous messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask something about the ontology...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        try:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                     output = await mcp_server.query_server(user_input)
                     for i in output:
                        st.markdown(i)
                        st.session_state.chat_history.append(
                         {"role": "assistant", "content": i}
                         )
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.session_state.chat_history.append(
                {"role": "assistant", "content": error_msg}
            )
            st.chat_message("assistant").write(error_msg)

    