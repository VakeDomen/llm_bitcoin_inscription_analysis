
use candle_core::{self, Device};

use crate::{
    config::DEFAULT_PROMPT, 
    llm::{
        model::load_model,
        prompt::{prompt_model, Prompt}, 
        tokenizer::load_tokenizer
    }
};

mod llm;
mod config;


fn main() {
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );

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

    let model = match load_model("models/llama3-8b/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf", &device) { 
        Ok(m) => m,
        Err(e) => panic!("Can't load model: {:#?}", e),
    };

    let prompt = Prompt::One(DEFAULT_PROMPT.to_string());

    match prompt_model(model, tokenizer, prompt, &device) {
        Ok(out) => println!("{}", out),
        Err(e) => panic!("Can't prompt model: {:#?}", e),
    }
    

}



