use candle_core::{quantized::gguf_file::Content, Device};
use candle_transformers::{generation::{LogitsProcessor, Sampling}, models::quantized_llama::ModelWeights};
use anyhow::Result;

use crate::config::{SEED, TEMPERATURE, TOP_K, TOP_P};

/// Loads model weights from a file path on a specified device.
///
/// # Arguments
/// * `model_path` - A string slice that specifies the path to the model file.
/// * `device` - A reference to the device (CPU, GPU) where the model will be loaded.
///
/// # Returns
/// A `Result` containing the loaded model weights or an error if the operation fails.
pub fn load_model(model_path: &str, device: &Device) -> Result<ModelWeights> {
    let model_path = std::path::PathBuf::from(model_path);
    let mut file = std::fs::File::open(&model_path)?;
    let start = std::time::Instant::now();
    
    let model = Content::read(&mut file).map_err(|e| e.with_path(model_path))?;
    
    println!(
        "Loaded {:?} tensors in {:.2}s",
        model.tensor_infos.len(),
        start.elapsed().as_secs_f32(),
    );
    
    let weights = ModelWeights::from_gguf(model, &mut file, device)?;
    Ok(weights)
}

/// Sets up a logits processor based on predefined settings and sampling strategy.
///
/// # Returns
/// An instance of `LogitsProcessor` configured with a specific sampling strategy.
pub fn setup_logit_procesing() -> LogitsProcessor {
    let sampling = setup_sampling();
    LogitsProcessor::from_sampling(SEED, sampling)
}

/// Configures the sampling strategy based on predefined temperature and probability settings.
///
/// # Returns
/// A `Sampling` variant configured according to the global temperature, TOP_K, and TOP_P settings.
fn setup_sampling() -> Sampling {
    if TEMPERATURE <= 0. {
        Sampling::ArgMax
    } else {
        match (TOP_K, TOP_P) {
            (None, None) => Sampling::All { temperature: TEMPERATURE },
            (Some(k), None) => Sampling::TopK { k, temperature: TEMPERATURE },
            (None, Some(p)) => Sampling::TopP { p, temperature: TEMPERATURE },
            (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p,temperature:  TEMPERATURE },
        }
    }
}