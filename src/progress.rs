use std::{fs::File, io::{BufWriter, Read, Write}};
use anyhow::Result;
use serde::{Serialize, Deserialize};

use crate::config::{INSCRIPTIONS_TO_PROCESS, PAR_CHUNK_SIZE};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Progress {
    pub batches_done: u64,
    pub par_chunk_size: u64,
    pub inscriptions_to_process: Option<usize>,
}

impl Default for Progress {
    fn default() -> Self {
        Self { 
            batches_done: 0, 
            par_chunk_size: PAR_CHUNK_SIZE, 
            inscriptions_to_process: INSCRIPTIONS_TO_PROCESS 
        }
    }
}

pub fn save_progress(progress: &Progress, file_path: &str) -> Result<()>{
    let file = File::create(file_path)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, progress)?;
    writer.flush()?;
    Ok(())
}

pub fn load_progress(file_path: &str) -> Progress {
    match File::open(file_path) {
        Ok(file) => progress_from_file(file),
        Err(e) => {
            println!("Failed loading progress: {}", e);
            println!("Reverting to defaults.");
            Progress::default()
        },
    }
}

fn progress_from_file(mut file: File) -> Progress {
    println!("Loading progress from file...");
    let mut data = String::new();
    match file.read_to_string(&mut data) {
        Ok(_) => {
            match serde_json::from_str(&data) {
                Ok(pro) => pro,
                Err(e) => {
                    println!("Error parsing progress file contents. File contents invalid: {}", e);
                    Progress::default()
                },
            }
        
        },
        Err(e) => {
            println!("Error reading progress file: {}", e);
            Progress::default()
        },
    }
}