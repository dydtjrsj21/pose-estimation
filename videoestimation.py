import cv2
from furiosa.models.vision import YOLOv7w6Pose
from furiosa.runtime.sync import create_runner
import os

def process_video(video_path, output_path, fps=30.0):
    # YOLOv7 Pose 모델 인스턴스 생성
    yolo_pose = YOLOv7w6Pose()
    
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
    
    # 비디오 속성 가져오기
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 비디오 writer 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # 런타임 생성 및 프레임 처리
    with create_runner(yolo_pose.model_source()) as runner:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # 전처리
                inputs, contexts = yolo_pose.preprocess([frame])
                
                # 추론
                output = runner.run(inputs)
                
                # 후처리
                results = yolo_pose.postprocess(output, contexts=contexts)
                
                # 결과 시각화
                if results and len(results[0]) > 0:
                    yolo_pose.visualize(frame, results[0])
                
                # 프레임 저장
                out.write(frame)
                
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                continue
    
    # 리소스 해제
    cap.release()
    out.release()
    print(f"Processing completed. Results saved to {output_path}")

if __name__ == "__main__":
    video_path = "tests/assets/pushup2.mp4"  # 입력 비디오 경로
    output_path = "output_video_pose.mp4"  # 출력 비디오 경로
    
    try:
        process_video(video_path, output_path)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
