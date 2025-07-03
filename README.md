# Sentiment Analysis in Rust

This project is a sentiment analysis model written in Rust. It uses a simple bag-of-words model with a linear layer, implemented using the `tch-rs` crate (Rust bindings for PyTorch).

## Requirements

*   Rust
*   `libtorch` (the C++ library for PyTorch)
*   A CUDA-enabled GPU is required.
*   The `IMDB Dataset.csv` file.

## Usage

This program provides two commands: `train` and `test`.

### Training the model

To train the model, run the following command:

```bash
cargo run --release -- train --batch-size 2048 --epochs 100 --learning-rate 1e-3
```

You can adjust the `batch-size`, `epochs`, and `learning-rate` as needed. The trained model will be saved as `model.ot` and the vocabulary as `vocab.json`.

### Testing the model

To test the model with a custom text, run the following command:

```bash
cargo run --release -- test "Your text to analyze here"
```

The model will output the sentiment (Positive/Negative) and the probability.

## Git LFS

This repository uses Git Large File Storage (LFS) to handle the large `IMDB Dataset.csv` file.

To work with this repository, you need to have Git LFS installed. You can install it from [https://git-lfs.github.com/](https://git-lfs.github.com/).

After installing Git LFS, you need to set it up for your user account by running:

```bash
git lfs install
```

Then, you can clone the repository as usual. Git LFS will automatically download the large file.
