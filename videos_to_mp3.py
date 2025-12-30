import os
import subprocess

files=os.listdir("videos")
for file in files:
    file_number=file.split(")")[0].split("(")[1]
    print(file_number)
    subprocess.run(
        [
            "ffmpeg",
            "-i", 
            f"videos/{file}",
            "-vn",
            f"audios/{file_number}_small.mp3"
        ]
    )