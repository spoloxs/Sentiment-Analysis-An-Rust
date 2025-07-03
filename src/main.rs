use clap::{Parser, Subcommand};
use csv::ReaderBuilder;
use rand::seq::SliceRandom;
use rand::thread_rng;
use regex::Regex;
use serde::Deserialize;
use std::collections::HashMap;
use std::error::Error;
use tch::{nn, nn::OptimizerConfig, Device, IndexOp, Kind, Tensor};
use rayon::prelude::*;
use std::fs::File;
use std::io::{Write, Read};

use libc::dlopen;
use std::ffi::CString;

use sysinfo::{System, SystemExt, ProcessExt, Pid};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train the sentiment analysis model
    Train {
        /// The batch size for training
        #[arg(long, default_value_t = 2048)]
        batch_size: usize,
        /// The number of epochs for training
        #[arg(long, default_value_t = 100)]
        epochs: i32,
        /// The learning rate for the optimizer
        #[arg(long, default_value_t = 1e-3)]
        learning_rate: f64,
    },
    /// Test the sentiment analysis model with a given text
    Test {
        /// The text to analyze
        text: String,
    },
}

#[derive(Debug, Deserialize)]
struct Record {
    review: String,
    sentiment: String,
}

fn clean_text(text: &str) -> String {
    let re = Regex::new(r"<[^>]*>").unwrap();
    re.replace_all(text, "").to_lowercase()
}

fn load_dataset(path: &str) -> Result<Vec<(String, u8)>, Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(path)?;
    let mut data = Vec::new();

    for result in rdr.deserialize() {
        let record: Record = result?;
        let cleaned = clean_text(&record.review);
        let label = if record.sentiment == "positive" { 1 } else { 0 };
        data.push((cleaned, label));
    }
    Ok(data)
}

struct Vectorizer {
    vocabulary: HashMap<String, usize>,
}

impl Vectorizer {
    fn new() -> Self {
        Vectorizer {
            vocabulary: HashMap::new(),
        }
    }

    fn fit(&mut self, texts: &[&str]) {
        let mut index = 0;
        for text in texts {
            for word in text.split_whitespace() {
                let word = word.to_lowercase();
                if !self.vocabulary.contains_key(&word) {
                    self.vocabulary.insert(word, index);
                    index += 1;
                }
            }
        }
    }

    fn transform<'a>(&self, texts: impl IntoParallelIterator<Item = &'a &'a str>) -> Vec<Vec<usize>> {
        texts
            .into_par_iter()
            .map(|text| {
                text.split_whitespace()
                    .filter_map(|word| self.vocabulary.get(&word.to_lowercase()))
                    .cloned()
                    .collect()
            })
            .collect()
    }
}

fn vectorize_batch(batch_indices: &[Vec<usize>], vocab_size: usize, device: Device) -> Tensor {
    let mut batch_tensors = Vec::new();
    for indices in batch_indices {
        let mut vec = vec![0f32; vocab_size];
        for &idx in indices {
            if idx < vocab_size {
                vec[idx] += 1.0;
            }
        }
        batch_tensors.push(Tensor::from_slice(&vec));
    }
    Tensor::stack(&batch_tensors, 0).to_device(device)
}

fn print_memory_usage(sys: &mut System) {
    sys.refresh_processes();
    let pid = Pid::from(std::process::id() as usize);
    if let Some(process) = sys.process(pid) {
        let mem = process.memory() as f64 / 1024f64.powi(3);
        println!("Memory usage: {} GB", mem);
    } else {
        println!("Could not get memory usage info.");
    }
}

fn train_model(device: Device, batch_size: usize, epochs: i32, learning_rate: f64) -> Result<(), Box<dyn Error>> {
    let mut sys = System::new_all();
    let path = "IMDB Dataset.csv";

    // Load and shuffle dataset
    let mut data = load_dataset(path)?;
    data.shuffle(&mut thread_rng());

    print_memory_usage(&mut sys);

    let split_idx = (data.len() as f32 * 0.8) as usize;
    let (train_data, test_data) = data.split_at(split_idx);

    let train_texts: Vec<&str> = train_data.iter().map(|(t, _)| t.as_str()).collect();
    let train_labels: Vec<u8> = train_data.iter().map(|(_, l)| *l).collect();

    let test_texts: Vec<&str> = test_data.iter().map(|(t, _)| t.as_str()).collect();
    let test_labels: Vec<u8> = test_data.iter().map(|(_, l)| *l).collect();

    // Vectorize
    let mut vectorizer = Vectorizer::new();
    vectorizer.fit(&train_texts);
    let vocab_size = vectorizer.vocabulary.len();

    // Save vocabulary
    let vocab_json = serde_json::to_string(&vectorizer.vocabulary)?;
    let mut file = File::create("vocab.json")?;
    file.write_all(vocab_json.as_bytes())?;
    println!("Vocabulary saved to vocab.json");

    // Pre-process texts into index sequences
    println!("Vectorizing training data into index sequences...");
    let x_train_indices = vectorizer.transform(&train_texts);
    let y_train = Tensor::from_slice(&train_labels)
        .to_device(device)
        .to_kind(Kind::Float)
        .view([-1, 1]);
    
    println!("Vectorizing test data into index sequences...");
    let x_test_indices = vectorizer.transform(&test_texts);
    let y_test = Tensor::from_slice(&test_labels)
        .to_device(device)
        .to_kind(Kind::Int64);

    print_memory_usage(&mut sys);

    // Model
    let vs = nn::VarStore::new(device);
    let root = &vs.root();
    let linear = nn::linear(root, vocab_size as i64, 1, Default::default());
    let mut opt = nn::Adam::default().build(&vs, learning_rate)?;

    // Train with batch learning
    let num_train_samples = train_data.len();
    let mut indices = (0..num_train_samples).collect::<Vec<_>>();

    for epoch in 1..=epochs {
        indices.shuffle(&mut thread_rng());

        let mut epoch_loss = 0.0;
        let mut batch_count = 0;

        for i in (0..num_train_samples).step_by(batch_size) {
            let batch_end = std::cmp::min(i + batch_size, num_train_samples);
            let batch_indices_for_slicing = &indices[i..batch_end];
            
            let current_batch_texts: Vec<_> = batch_indices_for_slicing.iter().map(|&idx| x_train_indices[idx].clone()).collect();
            let x_batch = vectorize_batch(&current_batch_texts, vocab_size, device);

            let y_batch_indices: Vec<i64> = batch_indices_for_slicing.iter().map(|&idx| idx as i64).collect();
            let y_batch = y_train.index_select(0, &Tensor::from_slice(&y_batch_indices).to_device(device));

            let logits = x_batch.apply(&linear).squeeze_dim(-1);
            let loss = logits.binary_cross_entropy_with_logits::<Tensor>(
                &y_batch.view([-1]),
                None,
                None,
                tch::Reduction::Mean,
            );

            opt.zero_grad();
            loss.backward();
            opt.step();

            epoch_loss += loss.double_value(&[]);
            batch_count += 1;
        }

        println!(
            "Epoch {}: Avg Loss = {:.4}",
            epoch,
            epoch_loss / batch_count as f64
        );
        print_memory_usage(&mut sys);
    }

    // Evaluate
    let mut total_correct = 0;
    let test_num_samples = test_data.len();

    for i in (0..test_num_samples).step_by(batch_size) {
        let batch_end = std::cmp::min(i + batch_size, test_num_samples);
        let batch_texts = &x_test_indices[i..batch_end];
        let x_test_batch = vectorize_batch(batch_texts, vocab_size, device);
        let y_test_batch = y_test.i((i as i64)..(batch_end as i64));

        let logits = x_test_batch.apply(&linear).squeeze_dim(-1);
        let preds = logits.sigmoid().ge(0.5).to_kind(Kind::Int64);
        let correct = preds.eq_tensor(&y_test_batch).sum(Kind::Int64);
        total_correct += correct.int64_value(&[]);
    }

    let accuracy = (total_correct as f64 / test_num_samples as f64) * 100.0;
    println!("Test Accuracy: {:.2}%", accuracy);
    print_memory_usage(&mut sys);

    // Save the model
    vs.save("model.ot")?;
    println!("Model saved to model.ot");

    Ok(())
}

fn test_model(device: Device, text: &str) -> Result<(), Box<dyn Error>> {
    // Load vocabulary
    let mut file = File::open("vocab.json")?;
    let mut vocab_json = String::new();
    file.read_to_string(&mut vocab_json)?;
    let vocabulary: HashMap<String, usize> = serde_json::from_str(&vocab_json)?;
    let vocab_size = vocabulary.len();
    let vectorizer = Vectorizer { vocabulary };

    // Load the model
    let mut vs = nn::VarStore::new(device);
    let linear = nn::linear(&vs.root(), vocab_size as i64, 1, Default::default());
    vs.load("model.ot")?;

    // Prepare the input text
    let cleaned_text = clean_text(text);
    let text_indices = vectorizer.transform(&[cleaned_text.as_str()]);
    let input_tensor = vectorize_batch(&text_indices, vocab_size, device);

    // Perform inference
    let logits = input_tensor.apply(&linear).squeeze_dim(-1);
    let prob = logits.sigmoid();
    let prediction = prob.double_value(&[]) > 0.5;

    println!("Text: '{}'", text);
    println!("Sentiment: {}", if prediction { "Positive" } else { "Negative" });
    println!("Probability: {:.4}", prob.double_value(&[]));

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let path =
        CString::new("/home/stoxy/.local/lib/python3.13/site-packages/torch/lib/libtorch_cuda.so")?;
    unsafe {
        if dlopen(path.as_ptr(), libc::RTLD_NOW).is_null() {
            eprintln!("Failed to load CUDA library with dlopen.");
            std::process::exit(1);
        }
    }
    
    if !tch::Cuda::is_available() {
        eprintln!("CUDA is not available. Exiting.");
        std::process::exit(1);
    }

    let device = Device::Cuda(0);
    println!("CUDA available: {}", tch::Cuda::is_available());
    println!("Using device: {:?}", device);

    let cli = Cli::parse();

    match &cli.command {
        Commands::Train { batch_size, epochs, learning_rate } => {
            println!("Starting model training...");
            train_model(device, *batch_size, *epochs, *learning_rate)?;
        }
        Commands::Test { text } => {
            println!("Testing model...");
            test_model(device, text)?;
        }
    }

    Ok(())
}
