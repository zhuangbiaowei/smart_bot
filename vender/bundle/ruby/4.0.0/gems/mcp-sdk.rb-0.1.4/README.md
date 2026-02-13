# MCP SDK for Ruby

[![Gem Version](https://badge.fury.io/rb/mcp-sdk.rb.svg)](https://badge.fury.io/rb/mcp-sdk.rb)

A Ruby implementation of the Model Context Protocol (MCP) for both connecting to MCP servers and creating MCP servers.

## Features

- **Client Support**: Connect to SSE (Server-Sent Events) and Stdio-based MCP servers
- **Server Support**: Create MCP servers with tool registration
- Type-safe client interfaces  
- Easy integration with Ruby applications
- Comprehensive error handling
- JSON-RPC 2.0 compliant

## Installation

Add this line to your application's Gemfile:

```ruby
gem 'mcp-sdk.rb'
```

And then execute:

```bash
$ bundle install
```

Or install it yourself as:

```bash
$ gem install mcp-sdk.rb
```

## Usage

### MCP Client

#### Connecting to an SSE-based MCP server

```ruby
require 'mcp-sdk.rb'
client = MCP::SSEClient.new('http://example.com/sse?key=api_key')
client.start
mcp_server_json = client.list_tools
puts JSON.pretty_generate(convertFormat(mcp_server_json))
```

#### Connecting to a Stdio-based MCP server

```ruby
require 'mcp-sdk.rb'

client = MCP::StdioClient.new('nodejs path/to/server_executable.js')
client.start
mcp_server_json = client.list_tools
puts JSON.pretty_generate(convertFormat(mcp_server_json))
```

### MCP Server

#### Creating an MCP Server

**Stdio Server (Default)**
```ruby
require 'mcp-sdk.rb'

# Create stdio server (processes JSON-RPC over stdin/stdout)
server = MCP::Server.new(
  name: "Demo",
  version: "1.0.0",
  type: "stdio"  # optional, this is the default
)

# Add an addition tool
server.add_tool("add") do |params|
  result = params["a"] + params["b"]
  {
    content: [{ type: "text", text: result.to_s }]
  }
end

# Start the server (listens on stdin, responds on stdout)
server.start
```

**SSE Server (HTTP with Server-Sent Events)**
```ruby
require 'mcp-sdk.rb'

# Create SSE server (HTTP server with SSE support)
server = MCP::Server.new(
  name: "Demo",
  version: "1.0.0",
  type: "sse",
  port: 8080
)

# Add tools as needed
server.add_tool("add") do |params|
  result = params["a"] + params["b"]
  {
    content: [{ type: "text", text: result.to_s }]
  }
end

server.add_tool("multiply") do |params|
  result = params["x"] * params["y"]
  {
    content: [{ type: "text", text: result.to_s }]
  }
end

server.add_tool("greet") do |params|
  name = params["name"] || "World"
  {
    content: [{ type: "text", text: "Hello, #{name}!" }]
  }
end

# Start the HTTP server
server.start
```

**Enhanced SSE Server (Advanced Features)**
```ruby
require 'mcp-sdk.rb'

# Create Enhanced SSE server with advanced features
server = MCP::Server.new(
  name: "Enhanced Demo",
  version: "1.0.0",
  type: "enhanced_sse",
  port: 8080
)

# Add tools as needed
server.add_tool("calculate") do |params|
  operation = params["operation"] || "add"
  a = params["a"] || 0
  b = params["b"] || 0
  
  result = case operation
  when "add" then a + b
  when "multiply" then a * b
  when "subtract" then a - b
  when "divide" then b != 0 ? a / b : "Error: Division by zero"
  else "Unknown operation"
  end
  
  {
    content: [{ type: "text", text: "#{a} #{operation} #{b} = #{result}" }]
  }
end

# Start the Enhanced SSE server
server.start
```

#### Server Types

**Stdio Server (`type: "stdio"`)**
- Default server type
- Communicates via stdin/stdout using JSON-RPC 2.0
- Perfect for command-line tools and process-based communication
- No additional configuration required

**SSE Server (`type: "sse"`)**
- HTTP server with Server-Sent Events support
- Requires `port` parameter
- Provides REST endpoints and real-time SSE communication
- Includes CORS support for web applications

**Enhanced SSE Server (`type: "enhanced_sse"`)**
- Advanced HTTP server with enhanced SSE capabilities
- Requires `port` parameter
- Provides all standard SSE features plus:
  - Connection management and tracking
  - Broadcasting to multiple clients
  - WebSocket-like bidirectional communication simulation
  - Enhanced health monitoring
  - Connection status endpoints
- Includes CORS support for web applications
- Built with Sinatra for better performance and extensibility

#### SSE Server Protocol

The MCP SSE server follows a specific two-step protocol flow:

**Step 1: Get Message Endpoint**
```bash
curl http://localhost:8080/sse
```
This returns the endpoint information in SSE format:
```
event: endpoint
data: /mcp/message
```

**Step 2: Send JSON-RPC to Message Endpoint**
```bash
curl -X POST http://localhost:8080/mcp/message \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
```
This returns the JSON-RPC response in SSE format:
```
data: {"jsonrpc":"2.0","id":1,"result":{"tools":[...]}}
```

**Additional Endpoints:**
- `GET /health` - Server health check (convenience)

#### Enhanced SSE Server Protocol

The Enhanced SSE server provides all standard SSE endpoints plus advanced features:

**Standard Endpoints:**
- `GET /sse` - Get message endpoint (MCP protocol compliance)
- `POST /mcp/message` - Send JSON-RPC requests and receive SSE responses
- `GET /health` - Enhanced health check with detailed server information

**Advanced Endpoints:**
- `GET /sse/events` - Advanced SSE endpoint with connection management
- `POST /mcp/broadcast` - Broadcast messages to all connected SSE clients
- `GET /ws/connect` - WebSocket-like connection simulation (long polling)
- `POST /ws/send/:connection_id` - Send messages to specific connections
- `GET /connections` - View active connection status

**Enhanced SSE Events Example:**
```bash
# Connect to advanced SSE endpoint
curl -N http://localhost:8080/sse/events
# Returns connection ID and keeps connection alive with heartbeats

# Broadcast to all connected clients
curl -X POST http://localhost:8080/mcp/broadcast \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'

# Check connection status
curl http://localhost:8080/connections
```

#### Server API

- `MCP::Server.new(name:, version:, type:, port:)` - Create a new server instance
  - `name`: Server name (required)
  - `version`: Server version (required)  
  - `type`: Server type - `"stdio"` (default), `"sse"`, or `"enhanced_sse"` (optional)
  - `port`: Port number (required for SSE servers, ignored for stdio)
- `server.add_tool(name, &block)` - Register a tool with a block that receives parameters
- `server.start` - Start the server and listen for requests
- `server.stop` - Stop the server
- `server.list_tools` - Get list of registered tools
- `server.call_tool(name, arguments)` - Call a tool directly

The server implements the MCP protocol over JSON-RPC 2.0, supporting:
- `tools/list` - List available tools
- `tools/call` - Execute a specific tool

#### Examples

**Test MCP SSE Protocol with curl:**
```bash
# Step 1: Get the message endpoint
curl http://localhost:8080/sse
# Returns: event: endpoint\ndata: /mcp/message

# Step 2: Send JSON-RPC requests to the message endpoint
curl -X POST http://localhost:8080/mcp/message \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'

curl -X POST http://localhost:8080/mcp/message \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"add","arguments":{"a":5,"b":3}}}'

# Health check (convenience)
curl http://localhost:8080/health
```

**Test Stdio Server:**
```bash
# Send JSON-RPC to stdin
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | ruby your_server.rb
```

#### Tool Response Format

Tools should return responses in MCP format:

```ruby
{
  content: [
    { type: "text", text: "your response text" }
  ]
}
```

For simple text responses, you can return any value and it will be automatically wrapped in the proper format.

## Example Files

This repository includes several example files to help you get started:

- `example_enhanced_sse.rb` - Complete Enhanced SSE server example with multiple tools
- `enhanced_sse_client.html` - HTML client demo for testing SSE connections
- `test_enhanced_sse.rb` - Integration test suite for Enhanced SSE functionality
- `demo_both_servers.rb` - Demonstration of both stdio and SSE servers
- `example_usage.rb` - Basic usage examples

### Running Examples

**Start Enhanced SSE Server:**
```bash
ruby example_enhanced_sse.rb
```
Then open `enhanced_sse_client.html` in your browser to test the connection.

**Run Integration Tests:**
```bash
ruby test_enhanced_sse.rb
```

**Test Basic SSE Protocol:**
```bash
# Terminal 1: Start server
ruby example_enhanced_sse.rb

# Terminal 2: Test endpoints
curl http://localhost:8081/health
curl http://localhost:8081/sse
curl -X POST http://localhost:8081/mcp/message \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/zhuangbiaowei/mcp-sdk.rb.

## License

The gem is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).