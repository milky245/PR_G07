# Data Acquisition and Preprocessing

In this section, we will discuss how to acquire frames from animated movies and preprocess the data for training.

## Extracting Frames from Movies

The script `movie_frame_parser.py` is used to extract frames from movie files. Here’s a summary of how it works:

1. **File and Directory Setup:**
   - The script identifies the root directory and the movie file name using `os.path`.

2. **Video Capture:**
   - It uses `cv2.VideoCapture` to read the movie file.
   - Frames are read in a loop using `capture.read()`.

3. **Frame Extraction:**
   - Every 72nd frame is saved as an image using `cv2.imwrite`.
   - The images are stored in a directory structure under `frames/frames_<movie_filename>`.

4. **Main Function:**
   - Iterates through all movie files in the `movies` directory and calls `extract_frames_from_movie`.

Here’s a brief look at the core function:
```python
def extract_frames_from_movie(file_name: str):
    # Initialize video capture and directories
    # Read frames and save every 72nd frame
    # Return the total number of extracted frames
```
## Data Preprocessing

The script `clean_dataset.py` handles various preprocessing tasks. Here’s a summary of its functionality:

1. **Removing Small Pictures:**
   - `remove_small_pictures()` removes images smaller than 92x92 pixels from the dataset.

2. **Increasing Resolution:**
   - `increase_resolution_getchu()` and `increase_resolution_celeba()` resize images to 256x256 pixels.
   - These functions ensure uniform image dimensions suitable for model training.

3. **Arranging Data:**
   - `arrange_data_correctly(folder)` ensures that all images in the specified folder are cropped and resized to 256x256 pixels, maintaining aspect ratios.

Here are the key function names and their purposes:
```python
def remove_small_pictures():
    # Removes images smaller than 92x92 pixels

def increase_resolution_getchu():
    # Resizes images in the 'getchu' dataset to 256x256 pixels

def increase_resolution_celeba():
    # Resizes and crops images in the 'celeba' dataset to 256x256 pixels

def arrange_data_correctly(folder):
    # Crops and resizes images in the specified folder to 256x256 pixels
```

By extracting frames from animated movies and preprocessing the images to ensure consistent dimensions and quality, we prepare a robust dataset for training the GAN model.
