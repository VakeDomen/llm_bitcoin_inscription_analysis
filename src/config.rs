pub const INSCRIPTIONS_TO_PROCESS: Option<usize> = None;
pub const PAR_CHUNK_SIZE: u64 = 2;
pub const PROGRESS_FILE: &str = "./data/progress.json";


pub const SYSTEM_MSG: &str = "Your task is to help us analyize bitcoin inscriptions. You will tell me a summary of the given text. And then give a list of up to 5 keywords that would describe the inscription and a language field for the language (code) of the inscription. You only respond with JSON.";

pub const SEED: u64 = 42;

pub const TEMPERATURE: f64 = 0.4;
pub const SAMPLE_LEN: usize = 1000;
pub const TOP_K: Option<usize> = None;
pub const TOP_P: Option<f64> = None;

pub const VERBOSE_PROMPT: bool = false;
pub const SPLIT_PROPMT: bool = false;
pub const REPEAT_PENALTY: f32 = 1.1;
pub const REPEAT_LAST_N: usize = 64;