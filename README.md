# PR_G07
Group project of pattern recognition course

**step 1: 生成数据集:**

确保你的目录结构没有改变，将电影文件（如 movie1.mp4）放入 PR_G07/dataset_generator/movies/ 目录中。如果没有 movies 目录，请创建一个。

在命令行中，导航到 Project/dataset_generator/ 目录，然后运行py文件：

```bash
cd path/to/Project/dataset_generator
python movie_frame_parser.py
```
该文件预期效果应该是将从 movies/ 目录中的每个电影文件提取帧，并将提取的帧保存到 frames/frames_<movie_name>/ 目录中。

请检查PR_G07/dataset_generator/movie_frame_parser.py的代码正确性，因为目前还没有测试。

**step 2: pretrain and train:**

等第一步成功后进行
