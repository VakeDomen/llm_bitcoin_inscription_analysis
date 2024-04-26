
use std::{sync::{Arc, Mutex}, vec};

use candle_core::{self, Device};
use csv::Writer;
use indicatif::{ProgressBar, ProgressStyle};
use anyhow::Result;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::{
        inscriptions::loader::load_jsonl_data, llm::{
        model::load_model,
        prompt::{prompt_model, Prompt}, 
        tokenizer::load_tokenizer
    }
};

mod llm;
mod config;
mod inscriptions;

type ProcessedInscription = (String, String);

fn main() {
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );

    println!("Loading inscriptions");
    let inscriptions = match load_jsonl_data("./data/text_inscriptions.txt") {
        Ok(i) => i,
        Err(e) => panic!("Error loading inscriptions: {:#?}", e),
    };

    let device1 = match Device::new_cuda(0) {
        Ok(cuda) => cuda,
        Err(e) => {
            println!("Error initializing CUDA device. Switching to CPU. Error: {:#?}", e);
            Device::Cpu
        },
    };

    let device2 = match Device::new_cuda(1) {
        Ok(cuda) => cuda,
        Err(e) => {
            println!("Error initializing CUDA device. Switching to CPU. Error: {:#?}", e);
            Device::Cpu
        },
    };

    let tokenizer = match load_tokenizer("models/llama3-8b/tokenizer.json") {
        Ok(t) => t,
        Err(e) => panic!("Can't load tokenizer: {:#?}", e),
    };

    let mut model1 = match load_model("models/llama3-8b/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf", &device1) { 
        Ok(m) => Arc::new(Mutex::new(m)),
        Err(e) => panic!("Can't load model: {:#?}", e),
    };

    let mut model2 = match load_model("models/llama3-8b/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf", &device2) { 
        Ok(m) => Arc::new(Mutex::new(m)),
        Err(e) => panic!("Can't load model: {:#?}", e),
    };

    let mut model3 = match load_model("models/llama3-8b/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf", &device1) { 
        Ok(m) => Arc::new(Mutex::new(m)),
        Err(e) => panic!("Can't load model: {:#?}", e),
    };

    let mut model4 = match load_model("models/llama3-8b/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf", &device2) { 
        Ok(m) => Arc::new(Mutex::new(m)),
        Err(e) => panic!("Can't load model: {:#?}", e),
    };

    let progress_bar = get_progress_bar(inscriptions.len());

    let mut processed: Vec<ProcessedInscription> = vec![];
    let mut failed: Vec<ProcessedInscription> = vec![];

    let chunk_size = 2;

    for batch in inscriptions.chunks(chunk_size) {

        
        let results: Vec<Result<(String, String), (String, String)>> = batch.par_iter().enumerate().map(|(index, inscription)| {
            let prompt = Prompt::One(inscription.content.clone());
            println!("START: {}", index);

            // Select the appropriate model and device based on the index
            let (mut model, device) = match index {
                0 => (model1.lock().unwrap(), &device1),
                1 => (model2.lock().unwrap(), &device2),
                2 => (model3.lock().unwrap(), &device1),
                _ => (model4.lock().unwrap(), &device2),  
            };

            // Process the prompt with the selected model and device
            match prompt_model(&mut *model, &tokenizer, prompt, device) {
                Ok(out) => {
                    println!("END: {}", index);
                    Ok((inscription.id.clone(), out))
                },
                Err(e) => {
                    println!("ERROR: {}", index);
                    Err((inscription.id.clone(), e.to_string()))
                }
            }
        }).collect();

        for result in results {
            match result {
                Ok((id, out)) => {
                    processed.push((id, out));
                    println!("{}", processed.last().unwrap().1);  // Print last processed output
                },
                Err((id, err)) => failed.push((id, err)),
            }
        }
        progress_bar.inc(chunk_size as u64); 
    }

    progress_bar.finish_with_message("Processing complete!");

    if let Err(e) = save_to_csv(processed, "./data/processed_inscriptions.csv") {
        println!("Failed saving records: {:#?}", e)
    };
    if let Err(e) = save_to_csv(failed, "./data/failed_inscriptions.csv") {
        println!("Failed saving records: {:#?}", e)
    };

}

fn get_progress_bar(len: usize) -> ProgressBar {
    let progress_bar = ProgressBar::new(len as u64);
    progress_bar.set_style(ProgressStyle::with_template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
        .unwrap()
        .progress_chars("##-"));
    progress_bar
}

fn save_to_csv(records: Vec<ProcessedInscription>, file_name: &str) -> Result<()> {
    let mut wtr = Writer::from_path(file_name)?;
    let mut successful_writes = 0;
    let mut failed_writes = 0;
    println!("Saving file: {}", file_name);
    let progress_bar = get_progress_bar(records.len());
    for record in records.into_iter() {
        match wtr.write_record(&[record.0, record.1]) {
            Ok(_) => successful_writes += 1,
            Err(_) => failed_writes += 1,
        };
        progress_bar.inc(1); 
    }
    wtr.flush()?;
    println!("Saved: {} \tFailed: {}", successful_writes, failed_writes);
    Ok(())
}