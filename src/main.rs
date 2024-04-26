
use candle_core::{self, Device};

use crate::{
    config::DEFAULT_PROMPT, inscriptions::loader::load_jsonl_data, llm::{
        model::load_model,
        prompt::{prompt_model, Prompt}, 
        tokenizer::load_tokenizer
    }
};

mod llm;
mod config;
mod inscriptions;


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

    println!("N: {} -> {:#?}", inscriptions.len(), inscriptions.get(0));

    let device = match Device::new_cuda(0) {
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

    let mut model = match load_model("models/llama3-8b/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf", &device) { 
        Ok(m) => m,
        Err(e) => panic!("Can't load model: {:#?}", e),
    };


    for inscription in inscriptions.into_iter() {
        let prompt = Prompt::One(DEFAULT_PROMPT.to_string());
    
        let answer = match prompt_model(&mut model, &tokenizer, prompt, &device) {
            Ok(out) => out,
            Err(e) => panic!("Can't prompt model: {:#?}", e),
        };
        
    }

}



