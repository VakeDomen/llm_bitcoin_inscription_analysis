use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Inscription {
    pub id: String,
    pub content: String,
    block_no: i64,
    number: i64,
    timestamp: i64,
    content_length: Option<i64>,
    value: Option<i64>,
}