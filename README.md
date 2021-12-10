# VINR

## Prev exps
### 1. 재현님구조
- t로 z 반영비율 결정
- 같은 t면 z의 같은 위치를 같은 비율로 반영함

### 2. z로 modulation
- z를 (0, 1)로 normalize
- z에서 modulation parameter output
- t를 LFF로 amplify후 modulation parameter로 modSIREN 통과 -> RGB output

### 3. Mod + LIIF