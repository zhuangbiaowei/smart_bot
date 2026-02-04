---
description: Get current weather information
always: false
requires:
  bins: []
  env: [OPENWEATHER_API_KEY]
---

# Weather Skill

Use this skill to get weather information for any location.

## Usage

Use the `web_fetch` tool to get weather data from OpenWeatherMap API:

```ruby
# Example API call
api_key = ENV["OPENWEATHER_API_KEY"]
url = "https://api.openweathermap.org/data/2.5/weather?q=#{city}&appid=#{api_key}&units=metric"
```

## Configuration

Set your OpenWeather API key in the environment:

```bash
export OPENWEATHER_API_KEY="your-api-key"
```
