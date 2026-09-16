# Recreate

Recreate is a local image-reconstruction workflow built around two scripts:

- `describe.py`: generate a text prompt that describes an input image.
- `create.py`: generate an image from a prompt file, stdin, or interactive input.

This software does not use an image-to-image process.
The model generating the image only have the text as the reference to generate the image.

Both steps run with local Hugging Face models and do not require a remote API.

## Setup

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

The first real run downloads the selected model weights. Defaults:

- Vision model: `Qwen/Qwen3-VL-8B-Instruct`
- Image model: `black-forest-labs/FLUX.2-klein-4B`

For NVIDIA GPUs, install the PyTorch build that matches your CUDA version first.

## 1) Describe: image -> prompt

```bash
python describe.py <path_to_input_image>
```

Default output prompt path:

- `<input_stem>_recreated.prompt.txt`

Example:

```bash
python describe.py input.png --vision-model Qwen/Qwen3-VL-8B-Instruct
```

Useful options:

- `--output PATH`
- `--print` (print to stdout instead of saving a prompt file)
- `--force`
- `--vision-model MODEL`
- `--max-size INT` (default `512`)
- `--device auto|cuda|mps|cpu`

## 2) Create: prompt -> image

```bash
python create.py <path_to_prompt_file>
```

You can also provide the prompt without a file:

```bash
echo "A cinematic portrait of a fox in snow" | python create.py --stdin
python create.py --ask
python create.py --ask-multi
```

With `--ask-multi`, the script keeps asking for prompts until you submit an empty one.
If an output filename already exists, it asks whether to overwrite (`y/N`); if not, it asks for a new filename.

Default output image path:

- If prompt is `name.prompt.txt`, output is `name.png`
- For `*_recreated.prompt.txt`, output becomes `*_recreated.png`
- For `--stdin` or `--ask` without `--output`, output is `generated.png`
- For `--ask-multi` without `--output`, output names are `generated.png`, `generated_002.png`, `generated_003.png`, ...

Example:

```bash
python create.py input_recreated.prompt.txt --image-model Tongyi-MAI/Z-Image-Turbo --width 768 --height 512
```

Useful options:

- `--output PATH`
- `--force`
- `--stdin`
- `--ask`
- `--ask-multi`
- `--image-model MODEL`
- `--seed INT`
- `--steps INT` (default `28`)
- `--guidance-scale FLOAT` (default `3.5`)
- `--width INT` and `--height INT` (must be multiples of `8`)
- `--device auto|cuda|mps|cpu`

Existing files are not overwritten unless `--force` is provided.

## 3) Loop: continuous prompt -> image -> prompt cycle

```bash
python loop.py <directory> [options]
```

This script implements an infinite feedback loop:
1. Reads `prompt_N.txt` files from the directory.
2. Generates `image_N.png` from each prompt.
3. Describes the generated image to create `prompt_(N+1).txt`.
4. Repeats until stopped (Ctrl+C).

Useful for exploring prompt evolution and image generation drift over multiple iterations.

Useful options:

- `--start-from INT` (default: highest numbered `prompt_N.txt` or `image_N.png` in the directory)
- `--start-file PATH` (copy a text or image file into an empty directory as `prompt_0.txt` or `image_0.png`)
- `--steps INT` (number of iterations before exiting; default: run indefinitely)
- `--image-model MODEL`
- `--vision-model MODEL`
- `--width INT` and `--height INT`
- `--device auto|cuda|mps|cpu`

## License

See [LICENSE](LICENSE).
