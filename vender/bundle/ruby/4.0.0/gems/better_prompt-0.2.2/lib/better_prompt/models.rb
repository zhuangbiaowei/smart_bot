require 'sequel'
module BetterPrompt
  module ORM
    class Model < Sequel::Model(ORM.db[:models])
      plugin :timestamps, update_on_create: true
    end

    class Prompt < Sequel::Model(ORM.db[:prompts])
      plugin :timestamps, update_on_create: true
    end

    class ModelCall < Sequel::Model(ORM.db[:model_calls])
      plugin :timestamps, update_on_create: true
    end

    class Response < Sequel::Model(ORM.db[:responses])
      plugin :timestamps, update_on_create: true
    end

    class Feedback < Sequel::Model(ORM.db[:feedback])
      plugin :timestamps, update_on_create: true
    end

    class Tag < Sequel::Model(ORM.db[:tags])
      plugin :timestamps, update_on_create: true
    end

    class PromptTag < Sequel::Model(ORM.db[:prompt_tags])
    end
  end
end