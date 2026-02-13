require "json"
require "sinatra/base"
require "thread"

module MCP
  # Enhanced SSE server implementation with Angelo-like features but using Sinatra
  class EnhancedSSEServer < Sinatra::Base
    attr_reader :mcp_server_instance, :sse_connections, :ws_connections
    
    def initialize(mcp_server_instance)
      @mcp_server_instance = mcp_server_instance
      @sse_connections = []
      @ws_connections = {}
      @connection_mutex = Mutex.new
      super()
    end
    
    configure do
      set :server, :puma
      set :bind, '0.0.0.0'
      set :logging, true
    end
    
    # Enable CORS for all routes
    before do
      headers 'Access-Control-Allow-Origin' => '*',
              'Access-Control-Allow-Methods' => 'GET, POST, OPTIONS',
              'Access-Control-Allow-Headers' => 'Content-Type'
    end
    
    # Handle OPTIONS requests for CORS
    options '*' do
      200
    end
    
    # Standard SSE endpoint - returns the message endpoint for subsequent requests
    get '/sse' do
      content_type 'text/event-stream'
      headers 'Cache-Control' => 'no-cache',
              'Connection' => 'keep-alive'
      
      # Send endpoint event as per MCP SSE protocol
      response = "event: endpoint\n"
      response += "data: /mcp/message\n\n"
      response
    end
    
    # Enhanced SSE endpoint with connection management
    get '/sse/events' do
      content_type 'text/event-stream'
      headers 'Cache-Control' => 'no-cache',
              'Connection' => 'keep-alive',
              'X-Accel-Buffering' => 'no'  # Disable nginx buffering
      
      # Create a unique connection ID
      connection_id = SecureRandom.hex(8)
      
      # Add connection to our list
      @connection_mutex.synchronize do
        @sse_connections << {
          id: connection_id,
          response: response,
          created_at: Time.now
        }
      end
      
      stream do |out|
        begin
          # Send initial endpoint event
          out << "event: endpoint\n"
          out << "data: /mcp/message\n\n"
          
          # Send connection established event
          out << "event: connected\n"
          out << "data: #{connection_id}\n\n"
          
          # Keep connection alive
          loop do
            sleep 30
            out << "event: heartbeat\n"
            out << "data: #{Time.now.to_i}\n\n"
          end
        rescue => e
          puts "SSE connection error: #{e.message}"
        ensure
          # Clean up connection
          @connection_mutex.synchronize do
            @sse_connections.reject! { |conn| conn[:id] == connection_id }
          end
        end
      end
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
    
    # Broadcast endpoint - sends messages to all connected SSE clients
    post '/mcp/broadcast' do
      content_type 'application/json'
      
      begin
        request_data = JSON.parse(request.body.read)
        response = @mcp_server_instance.send(:handle_request, request_data)
        
        # Broadcast to all connected SSE clients
        broadcasted_count = 0
        @connection_mutex.synchronize do
          @sse_connections.each do |connection|
            begin
              connection[:response] << "event: broadcast\n"
              connection[:response] << "data: #{response.to_json}\n\n"
              broadcasted_count += 1
            rescue => e
              puts "Broadcast error to connection #{connection[:id]}: #{e.message}"
            end
          end
        end
        
        { 
          status: 'broadcasted', 
          message: 'Message sent to all connected clients',
          clients: broadcasted_count,
          response: response
        }.to_json
      rescue JSON::ParserError => e
        error_response = {
          jsonrpc: "2.0",
          id: nil,
          error: {
            code: -32700,
            message: "Parse error: #{e.message}"
          }
        }
        
        # Broadcast error to all clients
        @connection_mutex.synchronize do
          @sse_connections.each do |connection|
            begin
              connection[:response] << "event: error\n"
              connection[:response] << "data: #{error_response.to_json}\n\n"
            rescue
              # Ignore broadcast errors
            end
          end
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
        
        # Broadcast error to all clients
        @connection_mutex.synchronize do
          @sse_connections.each do |connection|
            begin
              connection[:response] << "event: error\n"
              connection[:response] << "data: #{error_response.to_json}\n\n"
            rescue
              # Ignore broadcast errors
            end
          end
        end
        
        { status: 'error', message: e.message }.to_json
      end
    end
    
    # WebSocket simulation endpoint (using long polling)
    get '/ws/connect' do
      content_type 'text/event-stream'
      headers 'Cache-Control' => 'no-cache',
              'Connection' => 'keep-alive'
      
      connection_id = SecureRandom.hex(8)
      
      stream do |out|
        @ws_connections[connection_id] = out
        
        begin
          out << "event: ws_connected\n"
          out << "data: #{connection_id}\n\n"
          
          # Keep connection alive and handle incoming messages
          loop do
            sleep 1
            # In a real WebSocket implementation, this would handle bidirectional communication
            # For now, we just keep the connection alive
          end
        rescue => e
          puts "WebSocket simulation error: #{e.message}"
        ensure
          @ws_connections.delete(connection_id)
        end
      end
    end
    
    # Send message to WebSocket connection
    post '/ws/send/:connection_id' do
      connection_id = params[:connection_id]
      
      if @ws_connections[connection_id]
        begin
          request_data = JSON.parse(request.body.read)
          response = @mcp_server_instance.send(:handle_request, request_data)
          
          @ws_connections[connection_id] << "event: ws_message\n"
          @ws_connections[connection_id] << "data: #{response.to_json}\n\n"
          
          { status: 'sent', connection_id: connection_id }.to_json
        rescue => e
          { status: 'error', message: e.message }.to_json
        end
      else
        status 404
        { status: 'error', message: 'Connection not found' }.to_json
      end
    end
    
    # Health check endpoint with enhanced information
    get '/health' do
      content_type 'application/json'
      {
        status: 'ok',
        server: @mcp_server_instance.name,
        version: @mcp_server_instance.version,
        type: 'enhanced_sse',
        tools_count: @mcp_server_instance.tools.size,
        protocol: 'MCP SSE (Enhanced Sinatra)',
        endpoints: {
          sse: '/sse',
          sse_events: '/sse/events',
          message: '/mcp/message',
          broadcast: '/mcp/broadcast',
          ws_connect: '/ws/connect',
          ws_send: '/ws/send/:connection_id'
        },
        connections: {
          sse: @sse_connections.length,
          ws: @ws_connections.length
        },
        uptime: Time.now.to_i
      }.to_json
    end
    
    # Connection status endpoint
    get '/connections' do
      content_type 'application/json'
      {
        sse_connections: @sse_connections.map { |conn| 
          { 
            id: conn[:id], 
            created_at: conn[:created_at].iso8601,
            age_seconds: (Time.now - conn[:created_at]).to_i
          } 
        },
        ws_connections: @ws_connections.keys,
        total: @sse_connections.length + @ws_connections.length
      }.to_json
    end
  end
end