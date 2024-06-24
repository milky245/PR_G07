### Data Acquisition and Processing

In this section, we will discuss how data is acquired and processed for training the GAN model. This includes extracting frames from movies and preprocessing the images to prepare them for the model.

#### Extracting Frames from Movies

To create a dataset from movies, frames are extracted at regular intervals. For this project, the movies used are "Your Name" and "Weathering with You" by Makoto Shinkai.

1. **File and Directory Setup:**
   - The script identifies the root directory and the movie file name. It creates a directory structure to store the extracted frames.

2. **Video Capture:**
   - It uses `cv2.VideoCapture` to read the movie file. Frames are read in a loop using `capture.read()`.

3. **Frame Extraction:**
   - Every 72nd frame is saved as an image using `cv2.imwrite`. This interval ensures a reasonable number of frames are extracted without redundancy.
   - The images are stored in a directory named `frames/frames_<movie_filename>`.

4. **Main Function:**
   - Iterates through all movie files in the `movies` directory and calls `extract_frames_from_movie` for each movie.

This process ensures that a sufficient number of frames are extracted from the movies to create a diverse dataset.

#### Preprocessing the Dataset

The preprocessing script handles various tasks to clean and prepare the dataset for training. Here's a summary of its functionality:

1. **Removing Small Pictures:**
   - The script removes images smaller than 92x92 pixels from the dataset. These small images are likely not useful for training and can introduce noise.

2. **Arranging Data Correctly:**
   - The script processes images in a specified folder, cropping and resizing them to 256x256 pixels. This ensures that all images in the dataset have consistent dimensions, which is crucial for training the model.

#### Detailed Steps for Preprocessing

1. **Loading and Resizing Images:**
   - The images are read from the specified directories and resized to the target dimensions using `cv2.resize`.

2. **Cropping Images:**
   - For images that are not square, the script crops them to maintain the aspect ratio. This is done by calculating the midpoint and cropping accordingly.

3. **Saving Preprocessed Images:**
   - The processed images are saved in new directories, ensuring the original data remains intact for reference or further processing if needed.
