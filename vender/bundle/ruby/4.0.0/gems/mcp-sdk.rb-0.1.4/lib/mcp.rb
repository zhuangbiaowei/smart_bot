require File.expand_path("../mcp/types", __FILE__)
require File.expand_path("../mcp/client", __FILE__)
require File.expand_path("../mcp/stdio_client", __FILE__)
require File.expand_path("../mcp/sse_client", __FILE__)
require File.expand_path("../mcp/convert", __FILE__)
require File.expand_path("../mcp/server", __FILE__)

module MCP
  LATEST_PROTOCOL_VERSION = "2024-11-05".freeze
end