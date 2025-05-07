import json

def audit_map_characters(json_filepath):
    unique_chars = set()
    with open(json_filepath, "r", encoding="utf-8") as f:
        maps_data = json.load(f)
        for item in maps_data:
            map_text = item["map"]
            for char in map_text:
                unique_chars.add(char)
    return unique_chars

print("--- 학습 데이터 어휘 감사 ---")
train_chars = audit_map_characters("data/train_maps.json")
val_chars = audit_map_characters("data/val_maps.json")

all_data_chars = train_chars.union(val_chars)

print(f"학습 데이터셋 전체에서 발견된 고유 문자들: {all_data_chars}")
print(f"개수: {len(all_data_chars)}")

# 예상했던 맵 타일 외의 문자가 있는지 확인
expected_map_tiles = {'P', 'W', '-', 'T', 'G', 'O'} # 여기에 줄바꿈 '\n'도 포함해서 비교 가능
# <PROMPT>, <MAP> 태그는 제외하고 순수 맵 콘텐츠만

unexpected_chars = all_data_chars - expected_map_tiles - {'\n', ' '} # 일반 공백과 줄바꿈 제외
if '\n' not in all_data_chars:
    print("경고: 맵 데이터에 줄바꿈 문자('\\n')가 없는 것 같습니다. 맵이 한 줄로 되어있나요?")

print(f"예상 외 문자 (공백, 줄바꿈 제외): {unexpected_chars if unexpected_chars else '없음'}")

# 줄바꿈 문자를 보기 좋게 표현
printable_chars = [repr(c)[1:-1] for c in all_data_chars] # 예: '\n' -> '\\n'
print(f"발견된 고유 문자 (repr 형태로 표시): {sorted(list(printable_chars))}")