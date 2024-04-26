use std::io::Write;
use anyhow::{Error, Result};
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_llama::{ModelWeights, MAX_SEQ_LEN};
use tokenizers::Tokenizer;

use crate::{config::{
    REPEAT_LAST_N, REPEAT_PENALTY, SAMPLE_LEN, SPLIT_PROPMT, SYSTEM_MSG, VERBOSE_PROMPT
}, llm::model::setup_logit_procesing};

use super::tokenizer::TokenOutputStream;


#[derive(Debug)]
pub enum Prompt {
    One(String),
}

/// Parses a `Prompt` enum into a raw string format suitable for further processing.
/// 
/// # Arguments
/// * `prompt` - A reference to the `Prompt` enum.
///
/// # Returns
/// A `Result` containing the processed string or an error.
pub fn parse_prompt_to_raw(prompt: &Prompt) -> Result<String> {
    match &prompt {
        Prompt::One(prompt) => Ok(llama3_prompt(prompt.clone())),
    }
}


/// Formats a user message into a string structured for model processing.
/// 
/// # Arguments
/// * `user_msg` - The user message to format.
///
/// # Returns
/// A formatted string with predefined system and user sections.
pub fn llama3_prompt(user_msg: String) -> String {
    format!(
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", 
        SYSTEM_MSG,
        user_msg
    )
}


/// Generates model responses based on a given prompt using a specific tokenizer and model weights.
///
/// # Arguments
/// * `model` - Model weights used for generating responses.
/// * `tokenizer` - Tokenizer for encoding the prompt into tokens.
/// * `prompt` - The prompt provided by the user.
/// * `device` - The computation device (e.g., CPU, GPU) on which model inference is run.
///
/// # Returns
/// A `Result` containing the generated response string or an error.
pub fn prompt_model(
    mut model: ModelWeights, 
    tokenizer: Tokenizer, 
    prompt: Prompt, 
    device: &Device
) -> Result<String> {
    let mut response_chunks = vec![];
    let mut tos = TokenOutputStream::new(tokenizer);
   
    // Parse the prompt to a raw string format.
    let prompt_str = parse_prompt_to_raw(&prompt)?;
    print!("{}", &prompt_str);
    
    
    // Tokenize the prompt string for model processing.
    let tokens = tos
        .tokenizer()
        .encode(prompt_str, true)
        .map_err(anyhow::Error::msg)?;
    
    // Optionally, print each token and its ID if verbose logging is enabled.
    if VERBOSE_PROMPT {
        for (token, id) in tokens.get_tokens().iter().zip(tokens.get_ids().iter()) {
            let token = token.replace('▁', " ").replace("<0x0A>", "\n");
            println!("{id:7} -> '{token}'");
        }
    }
    
    // Handle token length restrictions by trimming if necessary.
    let prompt_tokens = tokens.get_ids();
    let to_sample = SAMPLE_LEN.saturating_sub(1);
    
    let prompt_tokens = if prompt_tokens.len() + to_sample > MAX_SEQ_LEN - 10 {
        let to_remove = prompt_tokens.len() + to_sample + 10 - MAX_SEQ_LEN;
        prompt_tokens[prompt_tokens.len().saturating_sub(to_remove)..].to_vec()
    } else {
        prompt_tokens.to_vec()
    };
    
    // Setup for generating model responses.
    let mut all_tokens = vec![];
    let mut logits_processor = setup_logit_procesing();

    let start_prompt_processing: std::time::Instant = std::time::Instant::now();
    let mut next_token = if !SPLIT_PROPMT {
        // Generate response in a single batch if not splitting.
        let input = Tensor::new(prompt_tokens.as_slice(), device)?.unsqueeze(0)?;
        let logits = model.forward(&input, 0)?;
        let logits = logits.squeeze(0)?;
        logits_processor.sample(&logits)?
    } else {
        // Generate response token by token if splitting.
        let mut next_token = 0;
        for (pos, token) in prompt_tokens.iter().enumerate() {
            let input = Tensor::new(&[*token], device)?.unsqueeze(0)?;
            let logits = model.forward(&input, pos)?;
            let logits = logits.squeeze(0)?;
            next_token = logits_processor.sample(&logits)?
        }
        next_token
    };
    let prompt_dt = start_prompt_processing.elapsed();
    all_tokens.push(next_token);
    
    
    // Collect chunks of the generated response.
    if let Some(token) = tos.next_token(next_token)? {
        let _ = flush_token(&token);
        response_chunks.push(token);
    }

    // Continue generating tokens until the sample length is reached or an end-of-sentence token is encountered.
    let eos_token = "<|eot_id|>";
    let eos_token = *tos.tokenizer().get_vocab(true).get(eos_token).unwrap();
    let start_post_prompt = std::time::Instant::now();
    let mut sampled = 0;
    for index in 0..to_sample {
        let input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
        let logits = model.forward(&input, prompt_tokens.len() + index)?;
        let logits = logits.squeeze(0)?;
        let logits = if REPEAT_PENALTY == 1. {
            logits
        } else {
            let start_at = all_tokens.len().saturating_sub(REPEAT_LAST_N);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                REPEAT_PENALTY,
                &all_tokens[start_at..],
            )?
        };
        next_token = logits_processor.sample(&logits)?;
        all_tokens.push(next_token);
        if let Some(token) = tos.next_token(next_token)? {
            let _ = flush_token(&token);
            response_chunks.push(token);
        }
        sampled += 1;
        if next_token == eos_token {
            break;
        };
    }
    
    let dt = start_post_prompt.elapsed();
    if VERBOSE_PROMPT {
        // Optionally print the final output and performance stats if verbose logging is enabled.
        if let Some(rest) = tos.decode_rest().map_err(Error::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
    
        println!(
            "\n\n{:4} prompt tokens processed: {:.2} token/s",
            prompt_tokens.len(),
            prompt_tokens.len() as f64 / prompt_dt.as_secs_f64(),
        );
        println!(
            "{sampled:4} tokens generated: {:.2} token/s",
            sampled as f64 / dt.as_secs_f64(),
        );
    }


    Ok(response_chunks.join(""))
}

/// Prints a token and ensures the output buffer is flushed, used primarily for verbose logging.
///
/// # Arguments
/// * `token` - The token to print.
///
/// # Returns
/// A `Result` indicating success or any error during flushing.
fn flush_token(token: &str) -> Result<()> {
    if VERBOSE_PROMPT {
        print!("{token}");
        std::io::stdout().flush()?;
    }
    Ok(())
}

