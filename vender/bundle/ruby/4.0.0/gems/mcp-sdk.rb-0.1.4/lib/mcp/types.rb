module MCP
  class Request
    def initialize(jsonrpc: nil, request_id: nil, method: nil, params: {})
      @jsonrpc = jsonrpc
      @id = request_id
      @method = method
      @params = params
    end

    def to_json
      {
        "jsonrpc": @jsonrpc,
        "id": @id,
        "method": @method,
        "params": @params,
      }.to_json
    end
  end

  class JSONRPCRequest < Request
    def initialize(request_id: nil, method: nil, params: {})
      super(jsonrpc: "2.0", request_id: request_id, method: method, params: params)
    end
  end

  class ListToolsRequest < JSONRPCRequest
    def initialize(request_id: 0)
      super(request_id: request_id, method: "tools/list")
    end
  end

  class CallMethodRequest < JSONRPCRequest
    def initialize(request_id: 0, params:)
      super(request_id: request_id, method: "tools/call", params: params)
    end
  end
end
