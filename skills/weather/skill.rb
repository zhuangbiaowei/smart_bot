# frozen_string_literal: true

# Weather Skill - 获取天气信息
# 类似于 OpenClaw 的 weather skill

SmartBot::Skill.register :weather do
  desc "获取当前天气和预报信息"
  ver "1.0.0"
  author_name "SmartBot Team"

  # 注册天气查询工具
  register_tool :get_weather do
    desc "Get current weather for a location"
    param_define :location, "City name or location", :string
    param_define :unit, "Temperature unit (c/f)", :string
    
    tool_proc do
      require "net/http"
      require "json"
      
      location = input_params["location"]
      unit = input_params["unit"] || "c"
      
      begin
        # 使用 wttr.in 免费天气 API（无需 API key）
        encoded_location = URI.encode_www_form_component(location)
        uri = URI("https://wttr.in/#{encoded_location}?format=j1")
        
        response = Net::HTTP.get_response(uri)
        
        if response.is_a?(Net::HTTPSuccess)
          data = JSON.parse(response.body)
          current = data["current_condition"].first
          
          temp_c = current["temp_C"]
          temp_f = current["temp_F"]
          temp = unit.downcase == "f" ? "#{temp_f}°F" : "#{temp_c}°C"
          
          {
            location: data["nearest_area"].first["areaName"].first["value"],
            country: data["nearest_area"].first["country"].first["value"],
            temperature: temp,
            condition: current["weatherDesc"].first["value"],
            humidity: "#{current["humidity"]}%",
            wind: "#{current["windspeedKmph"]} km/h",
            feels_like: unit.downcase == "f" ? "#{current["FeelsLikeF"]}°F" : "#{current["FeelsLikeC"]}°C"
          }
        else
          { error: "Weather service unavailable" }
        end
      rescue => e
        { error: "Failed to get weather: #{e.message}" }
      end
    end
  end

  # 注册天气预报工具
  register_tool :get_forecast do
    desc "Get weather forecast for a location"
    param_define :location, "City name or location", :string
    param_define :days, "Number of days (1-3)", :integer
    
    tool_proc do
      require "net/http"
      require "json"
      
      location = input_params["location"]
      days = [(input_params["days"] || 3), 3].min
      
      begin
        encoded_location = URI.encode_www_form_component(location)
        uri = URI("https://wttr.in/#{encoded_location}?format=j1")
        
        response = Net::HTTP.get_response(uri)
        
        if response.is_a?(Net::HTTPSuccess)
          data = JSON.parse(response.body)
          forecast = data["weather"].first(days).map do |day|
            {
              date: day["date"],
              max_temp: "#{day["maxtempC"]}°C / #{day["maxtempF"]}°F",
              min_temp: "#{day["mintempC"]}°C / #{day["mintempF"]}°F",
              condition: day["hourly"][12]["weatherDesc"].first["value"]
            }
          end
          
          {
            location: data["nearest_area"].first["areaName"].first["value"],
            forecast: forecast
          }
        else
          { error: "Weather service unavailable" }
        end
      rescue => e
        { error: "Failed to get forecast: #{e.message}" }
      end
    end
  end

  # 激活时的配置
  on_activate do
    SmartAgent.logger&.info "Weather skill activated!"
  end
end
