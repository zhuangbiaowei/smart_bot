require "json"
require "angelo"

module MCP
  # Angelo-based SSE server implementation for MCP
  class AngeloSSEServer < Angelo::Base
    attr_reader :mcp_server_instance
    
    def initialize(mcp_server_instance)
      @mcp_server_instance = mcp_server_instance
      super()
    end
    
    # Configure server settings
    def self.configure_for_mcp(port)
      port port
      addr "0.0.0.0"
    end
    
    # Enable CORS for all routes
    before do
      headers 'Access-Control-Allow-Origin' => '*',
              'Access-Control-Allow-Methods' => 'GET, POST, OPTIONS',
              'Access-Control-Allow-Headers' => 'Content-Type'
    end
    
    # Handle OPTIONS requests for CORS
    options '*' do
      halt 200
    end
    
    # SSE endpoint - returns the message endpoint for subsequent requests
    # This follows the MCP SSE protocol specification
    get '/sse' do
      content_type 'text/event-stream'
      headers 'Cache-Control' => 'no-cache',
              'Connection' => 'keep-alive'
      
      # Send endpoint event as per MCP SSE protocol
      response = "event: endpoint\n"
      response += "data: /mcp/message\n\n"
      response
    end
    
    # MCP message endpoint - handles POST requests and returns SSE responses
    post '/mcp/message' do
      content_type 'text/event-stream'
      headers 'Cache-Control' => 'no-cache',
              'Connection' => 'keep-alive'
      
      begin
        request_data = JSON.parse(request.body.read)
        response = @mcp_server_instance.send(:handle_request, request_data)
        
        # Return JSON-RPC response in SSE format
        sse_response = "data: #{response.to_json}\n\n"
        sse_response
      rescue JSON::ParserError => e
        error_response = {
          jsonrpc: "2.0",
          id: nil,
          error: {
            code: -32700,
            message: "Parse error: #{e.message}"
          }
        }
        "data: #{error_response.to_json}\n\n"
      rescue => e
        error_response = {
          jsonrpc: "2.0", 
          id: nil,
          error: {
            code: -32603,
            message: "Internal error: #{e.message}"
          }
        }
        "data: #{error_response.to_json}\n\n"
      end
    end
    
    # Alternative SSE endpoint using Angelo's eventsource functionality
    # This provides more advanced SSE features
    eventsource '/sse/events' do |sse|
      # Add the connection to the SSE stash for broadcasting
      sses << sse
      
      # Send initial endpoint event
      sse.event :endpoint, '/mcp/message'
      
      # Keep connection alive
      sse.on_close do
        puts "SSE client disconnected"
      end
    end
    
    # POST endpoint for broadcasting events to all SSE connections
    post '/mcp/broadcast' do
      content_type 'application/json'
      
      begin
        request_data = JSON.parse(request.body.read)
        response = @mcp_server_instance.send(:handle_request, request_data)
        
        # Broadcast to all connected SSE clients
        sses.each do |sse|
          sse.message response.to_json
        end
        
        { status: 'broadcasted', clients: sses.count }.to_json
      rescue JSON::ParserError => e
        error_response = {
          jsonrpc: "2.0",
          id: nil,
          error: {
            code: -32700,
            message: "Parse error: #{e.message}"
          }
        }
        
        sses.each do |sse|
          sse.message error_response.to_json
        end
        
        { status: 'error', message: e.message }.to_json
      rescue => e
        error_response = {
          jsonrpc: "2.0", 
          id: nil,
          error: {
            code: -32603,
            message: "Internal error: #{e.message}"
          }
        }
        
        sses.each do |sse|
          sse.message error_response.to_json
        end
        
        { status: 'error', message: e.message }.to_json
      end
    end
    
    # Health check endpoint
    get '/health' do
      content_type 'application/json'
      {
        status: 'ok',
        server: @mcp_server_instance.name,
        version: @mcp_server_instance.version,
        type: 'angelo_sse',
        tools_count: @mcp_server_instance.tools.size,
        protocol: 'MCP SSE (Angelo)',
        endpoints: {
          sse: '/sse',
          sse_events: '/sse/events',
          message: '/mcp/message',
          broadcast: '/mcp/broadcast'
        },
        connected_clients: sses.count
      }.to_json
    end
    
    # WebSocket endpoint for real-time bidirectional communication
    websocket '/ws' do |ws|
      websockets << ws
      
      ws.on_message do |msg|
        begin
          request_data = JSON.parse(msg)
          response = @mcp_server_instance.send(:handle_request, request_data)
          ws.write response.to_json
        rescue JSON::ParserError => e
          error_response = {
            jsonrpc: "2.0",
            id: nil,
            error: {
              code: -32700,
              message: "Parse error: #{e.message}"
            }
          }
          ws.write error_response.to_json
        rescue => e
          error_response = {
            jsonrpc: "2.0", 
            id: nil,
            error: {
              code: -32603,
              message: "Internal error: #{e.message}"
            }
          }
          ws.write error_response.to_json
        end
      end
      
      ws.on_close do
        puts "WebSocket client disconnected"
      end
    end
  end
end