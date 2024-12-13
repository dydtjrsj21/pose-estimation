import time
from furiosa.runtime.sync import create_runner
import numpy as np

# enf 파일 로드
with create_runner("yolov7-w6-pose.enf") as runner:
    # 더미 입력 데이터 생성
    dummy_input = np.random.randint(0, 255, size=(1,3,960,960), dtype=np.uint8)
    
    # 성능 측정
    latencies = []
    for i in range(100):
        start_time = time.time()
        output = runner.run(dummy_input)
        end_time = time.time()
        latencies.append(end_time - start_time)
    
    # 결과 분석
    avg_latency = sum(latencies) / len(latencies)
    throughput = 1.0 / avg_latency  # 초당 처리 가능한 프레임 수

    print(f"Average Latency: {avg_latency*1000:.2f} ms")
    print(f"Throughput: {throughput:.2f} FPS")
