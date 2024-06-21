import os
import cv2

def extract_frames_from_movie(file_name: str):
    root = os.path.dirname(__file__)
    print('Root directory:', root)

    capture = cv2.VideoCapture(file_name)
    file = os.path.basename(file_name)
    frame_count = 1
    total_frame_count = 1

    frames_path = os.path.join(root, f'frames/frames_{file}')
    print('Frames path:', frames_path)

    if not os.path.exists(frames_path):
        os.makedirs(frames_path)

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        if frame_count % 96 == 0:
            frame_path = os.path.join(frames_path, f'frame_{total_frame_count}.jpg')
            cv2.imwrite(frame_path, frame)
            total_frame_count += 1
        frame_count += 1

    capture.release()
    return total_frame_count

def main():
    movies_root = os.path.join(os.path.dirname(__file__), 'movies')
    print('Movies root directory:', movies_root)

    for movie in os.listdir(movies_root):
        print(f'Begin extracting frames for: {movie}')
        movie_path = os.path.join(movies_root, movie)
        extracted_frame_count = extract_frames_from_movie(movie_path)
        print(f'Done extracting frames for: {movie}\nExtracted frames: {extracted_frame_count}')
        print('--------------------')

if __name__ == '__main__':
    main()
