# Hugging Face release checklist

Use this checklist for every `HeartMuLa/MuLaCover` model release. Keep the
Hugging Face repository private until every required check below passes.

## 1. Prepare the release directory

- [ ] Start from a clean checkout of the intended Git tag or commit.
- [ ] Copy `release/huggingface/README.md` to the model repository as
  `README.md`.
- [ ] Copy the repository-root `MODEL_LICENSE` to the model repository as
  `MODEL_LICENSE`.
- [ ] Add `config.json`, `gen_config.json`, `tokenizer.json`,
  `model.safetensors.index.json`, and all five model shards.
- [ ] Do not include demo audio, test outputs, local paths, logs, secrets, or
  training-only artifacts.

## 2. Verify names and public wording

- [ ] Check all release files for consistent product naming and public-facing wording.
- [ ] All product references use `MuLaCover` or `mulacover` as appropriate.
- [ ] The code repository link points to `github.com/HeartMuLa/MuLaCover`.
- [ ] `contact@mulalabs.ai`, Discord, and the HeartMuLa paper are present.

## 3. Verify licenses

- [ ] `README.md` declares `license: cc-by-nc-4.0`.
- [ ] `MODEL_LICENSE` is byte-for-byte identical to the source-code repository:

  ```bash
  cmp MODEL_LICENSE /path/to/MuLaCover/MODEL_LICENSE
  ```

- [ ] The Model Card states that the weights and outputs are noncommercial
  unless MuLa Labs grants separate written authorization.
- [ ] The Apache-2.0 code license is not presented as the model-weight license.

## 4. Verify files and checksums

- [ ] The assembled directory contains exactly the expected release files.
- [ ] Regenerate checksums for the model payload:

  ```bash
  shasum -a 256 \
    tokenizer.json config.json gen_config.json model.safetensors.index.json \
    model-00001-of-00005.safetensors model-00002-of-00005.safetensors \
    model-00003-of-00005.safetensors model-00004-of-00005.safetensors \
    model-00005-of-00005.safetensors > SHA256SUMS
  ```

- [ ] Verify the generated manifest:

  ```bash
  shasum -a 256 -c SHA256SUMS
  ```

- [ ] Confirm that every shard named by `model.safetensors.index.json` exists,
  and that no unreferenced shard is present.

## 5. Validate before publishing

- [ ] Download the private repository into a fresh directory with
  `hf download`.
- [ ] Run one short smoke generation and one full-length generation using the
  documented MuLaCover commands.
- [ ] Record the Git commit, model commit, hardware, seed, inputs, duration,
  and output location in the internal release record.
- [ ] Review the rendered Model Card and verify every external link.
- [ ] Confirm the repository is still private.

## 6. Publish last

- [ ] Create the matching source-code tag and GitHub Release.
- [ ] Update `CHANGELOG.md`, `SECURITY.md`, the source README,
  and the Model Card so their version and release status agree.
- [ ] Obtain final MuLa Labs approval for the model license and release text.
- [ ] Change Hugging Face visibility only after all checks above pass.
- [ ] Download the public repository once more and re-run checksum validation.
