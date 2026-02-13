require "json"
require "sinatra/base"
require "puma"
require "securerandom"
require_relative "enhanced_sse_server"

module MCP
  class Server
    attr_reader :name, :version, :type, :port, :tools

    def initialize(name:, version:, type: "stdio", port: nil)
      @name = name
      @version = version
      @type = type.to_s
      @port = port
      @tools = {}
      @running = false
      
      validate_configuration
    end

    def add_tool(name, &block)
      unless block_given?
        raise ArgumentError, "Block required for tool '#{name}'"
      end
      
      @tools[name.to_s] = block
    end

    def start(io_in = $stdin, io_out = $stdout)
      @running = true
      
      case @type
      when "stdio"
        start_stdio_server(io_in, io_out)
      when "sse"
        start_sse_server
      when "enhanced_sse"
        start_enhanced_sse_server
      else
        raise ArgumentError, "Unknown server type: #{@type}"
      end
    end

    def stop
      @running = false
      @sse_server.stop if @sse_server
      @enhanced_server.stop if @enhanced_server
    end

    def list_tools
      tool_list = @tools.keys.map do |tool_name|
        {
          name: tool_name,
          description: "Tool: #{tool_name}",
          inputSchema: {
            type: "object",
            properties: {},
            required: []
          }
        }
      end

      {
        tools: tool_list
      }
    end

    def call_tool(name, arguments = {})
      tool_name = name.to_s
      
      unless @tools.key?(tool_name)
        raise Error, "Tool '#{tool_name}' not found"
      end

      begin
        result = @tools[tool_name].call(arguments)
        
        # Ensure result has the expected MCP format
        if result.is_a?(Hash) && result.key?(:content)
          result
        else
          # Wrap simple results in MCP format
          {
            content: [{ type: "text", text: result.to_s }]
          }
        end
      rescue => e
        raise Error, "Error executing tool '#{tool_name}': #{e.message}"
      end
    end

    private

    def validate_configuration
      case @type
      when "stdio"
        # No additional validation needed for stdio
      when "sse", "enhanced_sse"
        if @port.nil?
          raise ArgumentError, "Port is required for SSE server type"
        end
        unless @port.is_a?(Integer) && @port > 0 && @port < 65536
          raise ArgumentError, "Port must be a valid integer between 1 and 65535"
        end
      else
        raise ArgumentError, "Server type must be 'stdio', 'sse', or 'enhanced_sse'"
      end
    end

    def start_stdio_server(io_in, io_out)
      @io_in = io_in
      @io_out = io_out
      
      puts "MCP Server '#{@name}' v#{@version} (stdio) starting..."
      puts "Available tools: #{@tools.keys.join(', ')}"
      puts "Ready to accept JSON-RPC requests on stdin..."
      
      process_stdio_requests
    end

    def start_sse_server
      puts "MCP Server '#{@name}' v#{@version} (SSE) starting on port #{@port}..."
      puts "Available tools: #{@tools.keys.join(', ')}"
      
      app = create_sse_app
      @sse_server = Puma::Server.new(app)
      @sse_server.add_tcp_listener("0.0.0.0", @port)
      
      puts "SSE Server ready at http://localhost:#{@port}"
      puts "MCP SSE Protocol Endpoints:"
      puts "  GET /sse - Get message endpoint (returns 'event: endpoint\\ndata: /mcp/message')"
      puts "  POST /mcp/message - Send JSON-RPC requests and receive SSE responses"
      puts "  GET /health - Health check"
      
      @sse_server.run.join
    end

    def start_enhanced_sse_server
      puts "MCP Server '#{@name}' v#{@version} (Enhanced SSE) starting on port #{@port}..."
      puts "Available tools: #{@tools.keys.join(', ')}"
      
      # Create Enhanced SSE server app
      app = create_enhanced_sse_app
      @enhanced_server = Puma::Server.new(app)
      @enhanced_server.add_tcp_listener("0.0.0.0", @port)
      
      puts "Enhanced SSE Server ready at http://localhost:#{@port}"
      puts "MCP SSE Protocol Endpoints (Enhanced):"
      puts "  GET /sse - Get message endpoint (returns 'event: endpoint\\ndata: /mcp/message')"
      puts "  POST /mcp/message - Send JSON-RPC requests and receive SSE responses"
      puts "  GET /sse/events - Advanced SSE endpoint with connection management"
      puts "  POST /mcp/broadcast - Broadcast to all connected SSE clients"
      puts "  GET /ws/connect - WebSocket-like connection (long polling)"
      puts "  POST /ws/send/:id - Send message to specific connection"
      puts "  GET /health - Health check with detailed info"
      puts "  GET /connections - View active connections"
      
      # Start the Enhanced SSE server
      @enhanced_server.run.join
    end

    def process_stdio_requests
      while @running
        begin
          line = @io_in.gets
          break unless line
          
          line = line.strip
          next if line.empty?
          
          request = JSON.parse(line)
          response = handle_request(request)
          
          @io_out.puts response.to_json
          @io_out.flush
        rescue JSON::ParserError => e
          send_stdio_error_response(nil, -32700, "Parse error: #{e.message}")
        rescue => e
          send_stdio_error_response(nil, -32603, "Internal error: #{e.message}")
        end
      end
    end

    def create_sse_app
      server_instance = self
      connections = []
      connections_mutex = Mutex.new
      
      Sinatra.new do
        set :server, :puma
        set :bind, '0.0.0.0'
        set :port, server_instance.port
        
        # Enable CORS
        before do
          headers 'Access-Control-Allow-Origin' => '*',
                  'Access-Control-Allow-Methods' => ['GET', 'POST', 'OPTIONS'],
                  'Access-Control-Allow-Headers' => 'Content-Type'
        end
        
        options '*' do
          200
        end
        
        # SSE endpoint - returns the message endpoint for subsequent requests
        get '/sse' do
          content_type 'text/event-stream'
          headers 'Cache-Control' => 'no-cache',
                  'Connection' => 'keep-alive',
                  'X-Accel-Buffering' => 'no'
          
          stream(:keep_open) do |out|
            connections_mutex.synchronize do
              connections << out
            end
            
            # Send endpoint event as per MCP SSE protocol
            out << "event: endpoint\ndata: /mcp/message\n\n"            
            
            # Heartbeat timer to keep connection alive
            heartbeat = ::EventMachine.add_periodic_timer(15) do
              begin
                out << "event: ping\n"
                out << "data: #{Time.now.to_i}\n\n"
              rescue => e
                puts "Heartbeat error: #{e.message}"
              end
            end
            
            # Clean up when connection closes
            out.callback do
              connections_mutex.synchronize do
                connections.delete(out)
                ::EventMachine.cancel_timer(heartbeat)
              end
            end
            
            out.errback do
              connections_mutex.synchronize do
                connections.delete(out)
                ::EventMachine.cancel_timer(heartbeat)
              end
            end
          end
        end
        
        # MCP message endpoint - handles POST requests and returns SSE responses
        post '/mcp/message' do
          content_type 'text/event-stream'
          headers 'Cache-Control' => 'no-cache',
                  'Connection' => 'keep-alive',
                  'X-Accel-Buffering' => 'no'
          
          stream(:keep_open) do |out|
            begin
              request_data = JSON.parse(request.body.read)
              response = server_instance.send(:handle_request, request_data)
              
              # Return JSON-RPC response in SSE format
              out << "data: #{response.to_json}\n\n"
            rescue JSON::ParserError => e
              error_response = {
                jsonrpc: "2.0",
                id: nil,
                error: {
                  code: -32700,
                  message: "Parse error: #{e.message}"
                }
              }
              out << "data: #{error_response.to_json}\n\n"
            rescue => e
              error_response = {
                jsonrpc: "2.0",
                id: nil,
                error: {
                  code: -32603,
                  message: "Internal error: #{e.message}"
                }
              }
              out << "data: #{error_response.to_json}\n\n"
            ensure
              out.close
            end
          end
        end
        
        # Health check endpoint
        get '/health' do
          content_type 'application/json'
          {
            status: 'ok',
            server: server_instance.name,
            version: server_instance.version,
            type: server_instance.type,
            tools_count: server_instance.tools.size,
            protocol: 'MCP SSE',
            active_connections: connections.length,
            endpoints: {
              sse: '/sse',
              message: '/mcp/message'
            }
          }.to_json
        end
      end
    end

    def create_enhanced_sse_app
      server_instance = self
      sse_connections = []
      ws_connections = {}
      connection_mutex = Mutex.new
      
      Sinatra.new do
        set :server, :puma
        set :bind, '0.0.0.0'
        set :port, server_instance.port
        
        # Enable CORS
        before do
          headers 'Access-Control-Allow-Origin' => '*',
                  'Access-Control-Allow-Methods' => ['GET', 'POST', 'OPTIONS'],
                  'Access-Control-Allow-Headers' => 'Content-Type'
        end
        
        options '*' do
          200
        end
        
        # Standard SSE endpoint
        get '/sse' do
          content_type 'text/event-stream'
          headers 'Cache-Control' => 'no-cache',
                  'Connection' => 'keep-alive'
          
          response = "event: endpoint\ndata: /mcp/message\n\n"
          response
        end
        
        # Enhanced SSE endpoint with connection management
        get '/sse/events' do
          content_type 'text/event-stream'
          headers 'Cache-Control' => 'no-cache',
                  'Connection' => 'keep-alive',
                  'X-Accel-Buffering' => 'no'
          
          connection_id = SecureRandom.hex(8)
          
          stream do |out|
            connection_mutex.synchronize do
              sse_connections << {
                id: connection_id,
                response: out,
                created_at: Time.now
              }
            end
            
            begin
              out << "event: endpoint\n"
              out << "data: /mcp/message\n\n"
              
              out << "event: connected\n"
              out << "data: #{connection_id}\n\n"
              
              loop do
                sleep 30
                out << "event: heartbeat\n"
                out << "data: #{Time.now.to_i}\n\n"
              end
            rescue => e
              puts "SSE connection error: #{e.message}"
            ensure
              connection_mutex.synchronize do
                sse_connections.reject! { |conn| conn[:id] == connection_id }
              end
            end
          end
        end
        
        # MCP message endpoint
        post '/mcp/message' do
          content_type 'text/event-stream'
          headers 'Cache-Control' => 'no-cache',
                  'Connection' => 'keep-alive'
          
          begin
            request_data = JSON.parse(request.body.read)
            response = server_instance.send(:handle_request, request_data)
            
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
        
        # Broadcast endpoint - improved connection management
        post '/mcp/broadcast' do
          content_type 'application/json'
          
          begin
            request_data = JSON.parse(request.body.read)
            response = server_instance.send(:handle_request, request_data)
            
            broadcasted_count = 0
            failed_connections = []
            
            connection_mutex.synchronize do
              sse_connections.each do |connection|
                begin
                  connection[:response] << "event: broadcast\n"
                  connection[:response] << "data: #{response.to_json}\n\n"
                  broadcasted_count += 1
                rescue => e
                  puts "Broadcast error to connection #{connection[:id]}: #{e.message}"
                  failed_connections << connection[:id]
                end
              end
              
              # Clean up failed connections
              sse_connections.reject! { |conn| failed_connections.include?(conn[:id]) }
            end
            
            {
              status: 'broadcasted',
              message: 'Message sent to connected clients',
              clients: broadcasted_count,
              failed: failed_connections.length
            }.to_json
          rescue => e
            { status: 'error', message: e.message }.to_json
          end
        end
        
        # WebSocket simulation - improved with proper cleanup
        get '/ws/connect' do
          content_type 'text/event-stream'
          headers 'Cache-Control' => 'no-cache',
                  'Connection' => 'keep-alive',
                  'X-Accel-Buffering' => 'no'
          
          connection_id = SecureRandom.hex(8)
          
          stream(:keep_open) do |out|
            connection_mutex.synchronize do
              ws_connections[connection_id] = out
            end
            
            # Heartbeat for WebSocket simulation
            heartbeat = ::EventMachine.add_periodic_timer(15) do
              begin
                out << "event: ping\n"
                out << "data: #{Time.now.to_i}\n\n"
              rescue => e
                puts "WebSocket heartbeat error: #{e.message}"
              end
            end
            
            out << "event: ws_connected\n"
            out << "data: #{connection_id}\n\n"
            
            out.callback do
              connection_mutex.synchronize do
                ws_connections.delete(connection_id)
                ::EventMachine.cancel_timer(heartbeat)
              end
            end
            
            out.errback do
              connection_mutex.synchronize do
                ws_connections.delete(connection_id)
                ::EventMachine.cancel_timer(heartbeat)
              end
            end
          end
        end
        
        # Send to WebSocket connection
        post '/ws/send/:connection_id' do
          connection_id = params[:connection_id]
          
          connection_mutex.synchronize do
            if ws_connections[connection_id]
              begin
                request_data = JSON.parse(request.body.read)
                response = server_instance.send(:handle_request, request_data)
                
                ws_connections[connection_id] << "event: ws_message\n"
                ws_connections[connection_id] << "data: #{response.to_json}\n\n"
                
                { status: 'sent', connection_id: connection_id }.to_json
              rescue => e
                { status: 'error', message: e.message }.to_json
              ensure
                # Clean up failed connection
                ws_connections.delete(connection_id) if e
              end
            else
              status 404
              { status: 'error', message: 'Connection not found' }.to_json
            end
          end
        end
        
        # Enhanced health check
        get '/health' do
          content_type 'application/json'
          {
            status: 'ok',
            server: server_instance.name,
            version: server_instance.version,
            type: 'enhanced_sse',
            tools_count: server_instance.tools.size,
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
              sse: sse_connections.length,
              ws: ws_connections.length
            },
            uptime: Time.now.to_i
          }.to_json
        end
        
        # Connection status - improved thread safety
        get '/connections' do
          content_type 'application/json'
          
          connection_mutex.synchronize do
            {
              sse_connections: sse_connections.map { |conn|
                {
                  id: conn[:id],
                  created_at: conn[:created_at].iso8601,
                  age_seconds: (Time.now - conn[:created_at]).to_i
                }
              },
              ws_connections: ws_connections.keys,
              total: sse_connections.length + ws_connections.length
            }.to_json
          end
        end
      end
    end

    def handle_request(request)
      request_id = request["id"]
      method = request["method"]
      params = request["params"] || {}

      case method
      when "tools/list"
        {
          jsonrpc: "2.0",
          id: request_id,
          result: list_tools
        }
      when "tools/call"
        tool_name = params["name"]
        arguments = params["arguments"] || {}
        
        begin
          result = call_tool(tool_name, arguments)
          {
            jsonrpc: "2.0",
            id: request_id,
            result: result
          }
        rescue Error => e
          {
            jsonrpc: "2.0",
            id: request_id,
            error: {
              code: -32000,
              message: e.message
            }
          }
        end
      else
        {
          jsonrpc: "2.0",
          id: request_id,
          error: {
            code: -32601,
            message: "Method not found: #{method}"
          }
        }
      end
    end

    def send_stdio_error_response(request_id, code, message)
      response = {
        jsonrpc: "2.0",
        id: request_id,
        error: {
          code: code,
          message: message
        }
      }
      
      @io_out.puts response.to_json
      @io_out.flush
    end
  end
end