# G07 Group: GAN project for video art style transfer

Follow the steps below to set up your environment, prepare your datasets, and run the training and testing scripts.

## 1. Movie Files Directory

Ensure your movie files are placed in a directory named `movies`. The `movie_frame_parser.py` script will extract frames from these movies.

## 2. Extracting Frames from Movies

To extract frames from the movies:

1. Ensure the `movies` directory is in the same directory as `movie_frame_parser.py`.
2. Run the script to extract frames:
   ```bash
   python movie_frame_parser.py
   ```

   This script will read each movie file from the `movies` directory, extract every 72nd frame, and save them in a directory structure like `frames/frames_<movie_filename>`.

## 3. Dataset Preparation

Prepare your dataset directories and ensure they are structured correctly. Here are the required datasets:

- **Cartoon Dataset:** Extracted frames from movies like "Your Name" and "Weathering with You".
- **Landscape Dataset:** High-quality landscape images (e.g., LHQ dataset at 256 resolution).

The datasets should be placed in the `data` directory with the following structure:
```
data/
  (input_images)/
  (name_of_movie_frames)/
  test-v2/
```

## 4. Download and Place VGG19 Weights

Download the `vgg19.npy` file from the following link:
[VGG19 Weights](https://path-to-vgg19-npy-file)

Place the downloaded `vgg19.npy` file in the same directory as `vgg19.py`.

## 5. Training the Model

### Basic Training Command

To start the training process with default parameters, use the following command:
```bash
python main.py --mode train
```

### Training Command with Parameters

To customize the training parameters, you can specify additional arguments:
```bash
python main.py --mode train --image_size 256 --batch_size 16 --pre_train_iter 2000 --iter 20000 --learning_rate 1e-4 --save_dir 'saved_models' --train_out_dir 'train_output'
```

- `image_size`: Size of the input images (default is 256).
- `batch_size`: Number of images per batch (default is 16).
- `pre_train_iter`: Number of pre-training iterations (default is 5000).
- `iter`: Number of training iterations (default is 50000).
- `learning_rate`: Learning rate for the optimizer (default is 1e-4).
- `save_dir`: Directory to save the trained models.
- `train_out_dir`: Directory to save training output images.

## 6. Testing the Model

To test the trained model, use the following command:
```bash
python main.py --mode test
```

Ensure that the `test-v2` dataset is prepared in the `data` directory and contains the test images.

## Detailed Script Descriptions

### movie_frame_parser.py

This script extracts frames from movie files:

- **File and Directory Setup:** Identifies the root directory and movie file names.
- **Video Capture:** Reads the movie files using `cv2.VideoCapture`.
- **Frame Extraction:** Saves every 72nd frame as an image.
- **Main Function:** Iterates through all movie files and extracts frames.

### clean_dataset.py

This script handles various preprocessing tasks:

- **Removing Small Pictures:** Removes images smaller than 92x92 pixels.
- **Increasing Resolution:** Resizes images to 256x256 pixels.
- **Arranging Data:** Crops and resizes images to 256x256 pixels, maintaining aspect ratios.

### vgg19.py

This script sets up the VGG19 model:

- **Loading Pre-Trained Weights:** Loads weights from `vgg19.npy`.
- **Building the Model:** Constructs the VGG19 layers and sets up feature extraction.

### main.py

This script contains the main training and testing procedures:

- **Argument Parsing:** Sets up command-line arguments for configuration.
- **CartoonGAN Class:** Initializes parameters, sets up data, builds the model, and defines training/testing procedures.
- **Pre-Training:** Trains the generator using VGG19 perceptual loss before adversarial training.
- **Training:** Alternates between updating the generator and discriminator using adversarial loss.
- **Testing:** Loads the trained model and generates images for the test dataset.
