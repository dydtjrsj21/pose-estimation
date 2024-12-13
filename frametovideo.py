import cv2
import os
import glob

def frames_to_video(frame_dir, output_path, fps=24):
    # bmp 파일들을 정렬된 순서로 가져오기
    frames = sorted(glob.glob(os.path.join(frame_dir, '*.bmp')))
    
    if not frames:
        print("No frames found!")
        return
    
    # 첫 프레임으로 비디오 속성 설정
    frame = cv2.imread(frames[0])
    height, width, _ = frame.shape
    
    # 비디오 writer 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 각 프레임을 비디오에 쓰기
    for frame_path in frames:
        frame = cv2.imread(frame_path)
        out.write(frame)
    
    out.release()
    print(f"Video saved to {output_path}")

# 사용 예시
frame_dir = 'output_pose/_output/pushup2'  # bmp 파일들이 있는 디렉토리
output_path = 'output_video.mp4'   # 저장할 비디오 파일 경로
frames_to_video(frame_dir, output_path)
