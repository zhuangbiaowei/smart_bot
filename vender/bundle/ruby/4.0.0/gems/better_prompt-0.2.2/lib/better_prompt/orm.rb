require 'sequel'
module BetterPrompt
  module ORM
    class << self
      attr_reader :db

      def setup(database_url)
        raise "DATABASE_URL environment variable not set" unless database_url
        @db = Sequel.connect(database_url)
        @db.test_connection
      rescue => e
        raise "Failed to connect to database: #{e.message}"
      end
    end
  end
end