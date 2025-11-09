import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from dotenv import load_dotenv

from fastapi import FastAPI, Query
import uvicorn

load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
            
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        available_tools = [{ 
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        # Agentic loop - continue until we get a final text response
        final_response = []
        
        while True:
            # Call Claude API
            response = self.anthropic.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                messages=messages,
                tools=available_tools
            )

            # Add assistant's response to messages
            assistant_content = []
            tool_uses = []
            
            for content in response.content:
                assistant_content.append(content)
                if content.type == 'text':
                    final_response.append(content.text)
                elif content.type == 'tool_use':
                    tool_uses.append(content)
            
            messages.append({
                "role": "assistant",
                "content": assistant_content
            })
            
            # If no tool uses, we're done
            if not tool_uses:
                break
            
            # Execute all tool calls and collect results
            tool_results = []
            for tool_use in tool_uses:
                tool_name = tool_use.name
                tool_args = tool_use.input
                
                print(f"[Calling tool {tool_name} with args {tool_args}]")
                
                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result.content
                })
            
            # Add tool results to messages
            messages.append({
                "role": "user",
                "content": tool_results
            })
            
            # Check for stop reason to avoid infinite loops
            if response.stop_reason == "end_turn":
                break

        return "\n".join(final_response)

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


# Global client instance
mcp_client = MCPClient()
app = FastAPI()


@app.on_event("startup")
async def startup_event():
    """Initialize MCP client on server startup"""
    import sys
    if len(sys.argv) < 2:
        print("Warning: No server script path provided")
        print("Usage: python client.py <path_to_server_script>")
    else:
        server_script_path = sys.argv[1]
        await mcp_client.connect_to_server(server_script_path)
        print(f"\nFastAPI server starting on http://localhost:8000")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up MCP client on server shutdown"""
    await mcp_client.cleanup()


@app.get("/")
async def root():
    """Root endpoint with usage instructions"""
    return {
        "message": "MCP Client API",
        "usage": "Send GET request to /query?message=your_query_here",
        "example": "http://localhost:8000/query?message=Hello"
    }


@app.get("/query")
async def query_endpoint(message: str = Query(..., description="The query message to process")):
    """
    Process a query through the MCP client
    
    Args:
        message: The query string to process
        
    Returns:
        JSON response with the result
    """
    try:
        if not mcp_client.session:
            return {
                "error": "MCP client not connected. Please provide server script path.",
                "status": "error"
            }
        
        response = await mcp_client.process_query(message)
        return {
            "query": message,
            "response": response,
            "status": "success"
        }
    except Exception as e:
        return {
            "query": message,
            "error": str(e),
            "status": "error"
        }


async def main():
    import sys
    
    # Run FastAPI server
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    import sys
    asyncio.run(main())
