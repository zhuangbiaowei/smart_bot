require "faraday"
require "uri"
require "json"

module MCP
  class SSEClient < StdioClient
    def initialize(url, opt = {})
      @request_id = 0
      @pending_requests = {}
      @response_queue = Queue.new
      @running = false
      @endpoint = nil
      @temp_chunk = nil
      uri = URI(url)
      @conn = Faraday.new(url: uri.origin) do |f|
        f.adapter :net_http
        f.options.timeout = 30           # read timeout
        f.options.open_timeout = 10      # connection timeout
      end
      @thread = Thread.new do
        loop do
          begin
            @conn.get do |req|
              req.url uri.request_uri
              req.headers["Accept"] = "text/event-stream; charset=utf-8"
              req.headers["Accept-Encoding"] = "identity"
              req.headers["Content-Type"] = "application/json"

              req.options.on_data = proc do |chunk, overall_received_bytes|
                if @temp_chunk == nil
                  @temp_chunk = chunk
                else
                  @temp_chunk += chunk
                end
                event_type = @temp_chunk.split("\n")[0].split(":")[1].strip
                data = @temp_chunk.split("\n")[1][5..-1].strip
                if event_type == "endpoint"
                  @endpoint = data
                  @temp_chunk = nil
                else
                  if verify_json(data) && event_type == "message"
                    handle_response(data)
                    @temp_chunk = nil
                  end
                end
              end
            end
          rescue Faraday::Error => e
            #puts "SSE connection error: #{e.message}"
            sleep 1
          rescue => e
            #puts "Unexpected error in SSE thread: #{e.message}"
            sleep 1
          end
        end
      end
    end

    def verify_json(str)
      JSON.parse(str)
      true
    rescue JSON::ParserError
      false
    end

    def write_request(message)
      while (@endpoint == nil)
        sleep(0.2)
      end
      if @endpoint
        @conn.post(@endpoint, message, { "content-type" => "application/json" })
      end
    end

    def start
      request_id = next_id
      write_request({
        jsonrpc: "2.0",
        id: request_id,
        method: "initialize",
        params: {
          protocolVersion: "2025-04-28",
          clientInfo: { name: "mcp-ruby-client", version: "0.0.1" },
        },
      }.to_json)
      read_response(request_id)
      write_request({ jsonrpc: "2.0", method: "notifications/initialized" }.to_json)
    end

    def stop
    end
  end
end
