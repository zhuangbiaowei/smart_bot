require "json"
require "childprocess"
require "timeout"

module MCP
  class Error < StandardError; end

  class Client
    def initialize(server_cmd)
      cmd, server_path = server_cmd.split(" ")
      @server = ChildProcess.build(cmd, server_path)
      @stdout, @stdout_writer = IO.pipe
      @stderr, @stderr_writer = IO.pipe
      @server.io.stdout = @stdout_writer
      @server.io.stderr = @stderr_writer
      @server.duplex = true
      @request_id = 0
      @pending_requests = {}
      @response_queue = Queue.new
      @running = false
    end

    def start
      @server.start
      sleep 0.5 # Wait for server startup
      @running = true
      setup_io_handlers
    end

    def stop
      @running = false
      @stdout_writer.close
      @stderr_writer.close
      @reader.join if @reader
      @server.stop
      @stdout.close
      @stderr.close
    end

    def list_tools()
      request_id = next_id
      request = ListToolsRequest.new(request_id: request_id).to_json
      write_request(request)
      read_response(request_id)
    end

    def call_method(params)
      request_id = next_id
      request = CallMethodRequest.new(
        request_id: request_id,
        params: params,
      ).to_json
      write_request(request)
      read_response(request_id)
    end

    private

    def next_id
      @request_id += 1
    end

    def write_request(request)
      @server.io.stdin.puts request
    end

    def setup_io_handlers
      @reader = Thread.new do
        while @running
          begin
            if line = @stdout.gets
              handle_response(line)
            end
          rescue IOError => e
            break unless @running
            raise e
          end
        end
      end
    end

    def handle_response(line)
      response = JSON.parse(line)
      @pending_requests[response["id"]] = response
    rescue JSON::ParserError
      raise Error, "Invalid JSON response: #{line}"
    end

    def read_response(request_id, timeout = 10)
      Timeout.timeout(timeout) do
        while !@pending_requests.key?(request_id)
          sleep 0.1
        end
        response = @pending_requests.delete(request_id)
        process_response(response)
      end
    rescue Timeout::Error
      raise Error, "Timeout waiting for response"
    end

    def process_response(response)
      if response["error"]
        raise Error.new(response["error"]["message"])
      end
      response["result"]
    end
  end
end
