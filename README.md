# VINR
## 고려사항
- Patch를 정사각형이 아니게
- LIFF를 HR image z랑 loss를 줄지, rgb pixel까지 가서 줄지 -> 일단 구조는 z 생성

## Prev exps
### 1. 재현님구조
- t로 z 반영비율 결정
- 같은 t면 z의 같은 위치를 같은 비율로 반영함

### 2. z로 modulation
- z를 (0, 1)로 normalize
- z에서 modulation parameter output
- t를 LFF로 amplify후 modulation parameter로 modSIREN 통과 -> RGB output

### 3. Mod + LIIF
- lr z에서 hr z를 생성
- lr z와 coord로 바로 rgb 생성
