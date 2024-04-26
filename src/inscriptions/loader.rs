use std::{fs::File, io::{BufRead, BufReader}};

use anyhow::Result;

use crate::inscriptions::inscription::Inscription;

pub fn load_jsonl_data(file_name: &str) -> Result<Vec<Inscription>> {
    let file = File::open(file_name)?;
    let reader = BufReader::new(file);
    let mut inscriptions = vec![];
    for line in reader.lines() {
        let line = line?;
        let inscription: Inscription = serde_json::from_str(&line)?;
        inscriptions.push(inscription)
    }
    Ok(inscriptions)
}