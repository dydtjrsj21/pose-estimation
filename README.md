## file description

- **frametovideo.py** : 최종 결과가 bmp파일로 나오는데 이를 취합해서 video로 만듦
- **inference100.py** : 만든 모델에 대해서만 성능을 파악하기 위한 코드 평균 latency와 throughput측정(모델 자체의 성능만을 보기 위해서 간단한 더미 데이터를 input)
- **pose_estimation_model.yaml** : config파일을 기존 warboy-vision-model코드와 호환이 가능하도록 수정