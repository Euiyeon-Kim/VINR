# VINR

## Intuition
- RGB를 생성하는 것 때문에 성능이 떨어지는게 아니라 Pixel마다 z를 뽑는 것 때문에 성능이 떨어지는 느낌
- T로 sampling 하고 motion을 모델링 하기 위해서는 적어도 3장 이상 써야할 듯
- 구조 자체는 여러 frame을 받을 수 있도록 모델링 하는게 좋을 것 같음

## 고려사항
- Patch를 정사각형이 아니게
- LIFF를 HR image z랑 loss를 줄지, rgb pixel까지 가서 줄지 -> 일단 구조는 z 생성
- Smoothing loss in generated flow

## Prev exps
### 1. 재현님구조
- t로 z 반영비율 결정
- 같은 t면 z의 같은 위치를 같은 비율로 반영함

### 2. z로 modulation --> RGB 생성
- z를 (0, 1)로 normalize
- z에서 modulation parameter output
- t를 LFF로 amplify후 modulation parameter로 modSIREN 통과 -> RGB output

### 3. Mod + LIIF
- lr z에서 hr z를 생성
- lr z와 coord로 바로 rgb 생성
- Nan loss 떠서 학습 안됨

### 4. z로 modulation --> Flow 생성
- masking으로 선택만해도 본전 찾음
- Flow를 생성을 제대로 안함 **지금 구조에서는 못하는게 맞는 것 같다**