---
name: weather
description: 获取天气信息和预报。支持查询当前天气和未来天气预报。
triggers:
  - weather
  - 天气
  - 气温
  - 温度
  - forecast
  - 预报
  - 下雨
  - 晴天
---

# Weather Skill

获取天气信息和预报。

## Usage

```ruby
# 获取当前天气
SmartAgent::Tool.call(:get_weather, { "location" => "Shanghai", "unit" => "c" })

# 获取天气预报
SmartAgent::Tool.call(:get_forecast, { "location" => "Shanghai", "days" => 3 })
```

## CLI Usage

```bash
smart_bot agent -m "上海天气"
smart_bot agent -m "北京明天天气怎么样"
```

## API

此 skill 使用 wttr.in API（免费，无需 API Key）。
