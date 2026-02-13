-- 创建模型表
CREATE TABLE models (
    model_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_provider TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 创建提示词表
CREATE TABLE prompts (
    prompt_id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL,
    prompt_template_name TEXT NOT NULL,
    prompt_title TEXT,
    prompt_content TEXT NOT NULL,
    prompt_length INTEGER NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 创建调用记录表
CREATE TABLE model_calls (
    call_id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_list TEXT NOT NULL,
    model_id INTEGER,
    is_streaming INTEGER NOT NULL, -- 在SQLite中使用0/1代替布尔值
    temperature REAL,
    max_tokens INTEGER,
    top_p REAL,
    top_k INTEGER,
    additional_parameters TEXT, -- 使用JSON字符串存储
    call_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_id) REFERENCES models(model_id)
);

-- 创建响应表
CREATE TABLE responses (
    response_id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id INTEGER,
    response_content TEXT NOT NULL,
    response_length INTEGER NOT NULL,
    response_time_ms INTEGER NOT NULL, -- 响应时间(毫秒)
    is_streaming INTEGER NOT NULL, -- 在SQLite中使用0/1代替布尔值
    token_count INTEGER,               -- 令牌数量
    cost REAL,                         -- 调用成本
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (call_id) REFERENCES model_calls(call_id)
);

-- 创建用户评价表
CREATE TABLE feedback (
    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
    response_id INTEGER,
    rating INTEGER CHECK (rating BETWEEN 1 AND 5), -- 1-5星评价
    feedback_comment TEXT,
    accuracy_score INTEGER CHECK (accuracy_score BETWEEN 1 AND 10),
    relevance_score INTEGER CHECK (relevance_score BETWEEN 1 AND 10),
    creativity_score INTEGER CHECK (creativity_score BETWEEN 1 AND 10),
    helpfulness_score INTEGER CHECK (helpfulness_score BETWEEN 1 AND 10),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (response_id) REFERENCES responses(response_id)
);

-- 创建标签表，用于更好地分类提示词
CREATE TABLE tags (
    tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
    tag_name TEXT NOT NULL UNIQUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 创建提示词-标签关联表
CREATE TABLE prompt_tags (
    prompt_id INTEGER,
    tag_id INTEGER,
    PRIMARY KEY (prompt_id, tag_id),
    FOREIGN KEY (prompt_id) REFERENCES prompts(prompt_id),
    FOREIGN KEY (tag_id) REFERENCES tags(tag_id)
);

-- 创建索引以提高查询性能
CREATE INDEX idx_responses_call_id ON responses(call_id);
CREATE INDEX idx_feedback_response_id ON feedback(response_id);
CREATE INDEX idx_prompt_tags_prompt_id ON prompt_tags(prompt_id);