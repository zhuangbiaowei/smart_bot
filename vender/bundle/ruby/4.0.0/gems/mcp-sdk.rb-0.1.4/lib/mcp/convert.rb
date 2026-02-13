def convertFormat(mcp_server_json)
  toolCalling_json = {}
  toolCalling_json["tools"] ||= []
  mcp_server_json["tools"].each do |tool|
    function = {}
    function["name"] = tool["name"]
    function["description"] = tool["description"]
    function["parameters"] = {}
    function["parameters"]["type"] = "object"
    function["parameters"]["required"] = tool["inputSchema"]["required"] || []
    tool["inputSchema"]["properties"].each do |prop_name, prop_def|
      param_def = {
        type: prop_def["type"],
        description: prop_def["description"],
      }

      if prop_def["enum"]
        param_def["enum"] = prop_def["enum"]
      end
      function["parameters"]["properties"] ||= {}
      function["parameters"]["properties"][prop_name] = param_def
    end
    toolCalling_json["tools"] << {
      "type": "function",
      "function": function,
    }
  end
  toolCalling_json
end
