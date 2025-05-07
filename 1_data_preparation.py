import os  # 운영 체제와 상호 작용하기 위한 모듈입니다. 파일 경로 확인, 디렉토리 생성 등에 사용됩니다.
import random  # 무작위 작업(예: 데이터 섞기)을 위한 모듈입니다.
import json  # JSON 데이터를 다루기 위한 모듈입니다. 데이터를 파일로 저장하거나 불러올 때 사용됩니다.
from pathlib import Path  # 파일 및 디렉토리 경로를 객체 지향적으로 다루기 위한 모듈입니다.

# 맵 데이터 파일을 읽어오고, 맵 타일 표현을 변경하는 함수입니다.
# 각 맵 파일에는 <PROMPT>와 <MAP> 태그가 포함될 수 있으며, </MAP> 태그는 항상 존재한다고 가정합니다.
def load_map_data(map_dir):
    """
    지정된 디렉토리에서 'map_*.txt' 패턴의 맵 데이터 파일들을 읽어옵니다.
    각 파일에서 프롬프트와 맵 데이터를 추출하고, 특정 맵 타일 문자를 새로운 표현으로 변환합니다.

    Args:
        map_dir (str): 맵 파일들이 저장된 디렉토리 경로입니다.

    Returns:
        list: 각 요소가 프롬프트와 변환된 맵 데이터를 담고 있는 딕셔너리들의 리스트입니다.
              오류 발생 또는 유효한 데이터를 찾지 못한 경우 빈 리스트를 반환할 수 있습니다.
    """
    # 지정된 디렉토리에서 "map_"으로 시작하고 ".txt"로 끝나는 모든 파일의 목록을 가져옵니다.
    map_files = list(Path(map_dir).glob("map_*.txt"))
    all_maps = []  # 읽어온 모든 맵 데이터를 저장할 리스트입니다.

    # 맵 타일을 치환하기 위한 규칙을 정의한 딕셔너리입니다.
    # 예를 들어, 'P'는 '[P]'로, '-'는 '[-]' 등으로 변경됩니다.
    tile_replacement_map = {
        "P": "[P]", "W": "[W]", "-": "[-]",
        "T": "[T]", "G": "[G]", "O": "[O]",
    }

    # 찾은 각 맵 파일에 대해 반복 작업을 수행합니다.
    for map_file in map_files:
        with open(map_file, "r", encoding="utf-8") as f: # 파일을 읽기 모드, UTF-8 인코딩으로 엽니다.
            content = f.read()  # 파일 전체 내용을 문자열로 읽어옵니다.
            
            prompt = ""  # 프롬프트 내용을 저장할 변수입니다.
            map_data_original = "" # <MAP>과 </MAP> 사이의 원본 맵 데이터를 저장할 변수입니다.

            # 파일 내용에서 프롬프트와 맵 데이터를 구분하는 태그들을 정의합니다.
            prompt_start_tag = "<PROMPT>"
            prompt_end_tag = "</PROMPT>"
            map_start_tag = "<MAP>"
            map_end_tag = "</MAP>" # 이제 항상 존재한다고 가정합니다.

            # 프롬프트 태그(<PROMPT> ... </PROMPT>)를 찾아 해당 내용을 추출합니다.
            idx_prompt_start = content.find(prompt_start_tag)
            idx_prompt_end = content.find(prompt_end_tag)
            if idx_prompt_start != -1 and idx_prompt_end != -1 and idx_prompt_start < idx_prompt_end:
                prompt = content[idx_prompt_start + len(prompt_start_tag):idx_prompt_end].strip()
            
            # 맵 데이터 태그(<MAP> ... </MAP>)를 찾아 해당 내용을 추출합니다.
            idx_map_start = content.find(map_start_tag)
            idx_map_end = content.find(map_end_tag)
            if idx_map_start != -1 and idx_map_end != -1 and idx_map_start < idx_map_end:
                # <MAP> 태그와 </MAP> 태그 사이의 내용을 원본 맵 데이터로 저장합니다.
                map_data_original = content[idx_map_start + len(map_start_tag):idx_map_end].strip()
            else:
                # <MAP> 또는 </MAP> 태그가 없거나 순서가 잘못된 경우, 또는 프롬프트만 있고 맵 태그가 없는 경우에 대한 처리입니다.
                if content.strip() and not prompt and not (map_start_tag in content or map_end_tag in content) :
                    # 파일에 태그가 전혀 없고 내용만 있는 경우, 전체 내용을 맵 데이터로 간주합니다. (프롬프트는 없음)
                    print(f"정보: {map_file} 파일에 태그가 없습니다. 전체 내용을 맵 데이터로 사용합니다.")
                    map_data_original = content.strip()
                elif map_start_tag in content and map_end_tag not in content:
                    # <MAP> 태그는 있지만 </MAP> 태그가 없는 경우, 경고를 출력하고 이 파일은 건너뜁니다.
                    print(f"경고: {map_file} 파일에 <MAP> 태그는 있으나 </MAP> 태그가 없습니다. 이 파일을 건너뜁니다.")
                    continue # 다음 파일로 넘어갑니다.
                else:
                    # 그 외 태그 형식이 올바르지 않거나 맵 데이터를 찾을 수 없는 경우, 경고를 출력하고 건너뜁니다.
                    print(f"경고: {map_file} 파일의 태그 형식이 올바르지 않거나 맵 데이터를 찾을 수 없습니다. 이 파일을 건너뜁니다.")
                    continue # 다음 파일로 넘어갑니다.
            
            # 추출된 원본 맵 데이터와 프롬프트가 모두 비어있다면, 유효한 데이터가 없는 것으로 간주하고 건너뜁니다.
            # 이는 빈 파일이거나 태그만 있고 내용이 없는 파일을 처리하기 위함입니다.
            if not map_data_original.strip() and not prompt.strip():
                print(f"경고: {map_file} 파일에서 프롬프트와 맵 데이터를 모두 추출하지 못했습니다. 건너뜁니다.")
                continue # 다음 파일로 넘어갑니다.

            # 원본 맵 데이터에 대해 정의된 `tile_replacement_map` 규칙에 따라 문자열 치환 작업을 수행합니다.
            # 예를 들어, 맵 데이터 내의 'P'는 '[P]'로, 'W'는 '[W]' 등으로 변경됩니다.
            map_data_transformed = map_data_original
            for original_char, new_representation in tile_replacement_map.items():
                map_data_transformed = map_data_transformed.replace(original_char, new_representation)
            
            # 추출된 프롬프트와 변환된 맵 데이터를 딕셔너리 형태로 `all_maps` 리스트에 추가합니다.
            all_maps.append({
                "prompt": prompt,
                "map": map_data_transformed
            })
            
    return all_maps # 모든 파일 처리가 끝난 후, 맵 데이터 리스트를 반환합니다.

# 불러온 전체 맵 데이터를 학습용 데이터와 검증용 데이터로 나누는 함수입니다.
def split_train_val(all_maps, val_ratio=0.2):
    """
    전체 맵 데이터 리스트를 주어진 비율에 따라 학습용과 검증용으로 나눕니다.
    데이터는 나누기 전에 무작위로 섞입니다.

    Args:
        all_maps (list): 프롬프트와 맵 데이터를 담은 딕셔너리들의 리스트입니다.
        val_ratio (float, optional): 전체 데이터 중 검증용으로 사용할 비율입니다. 기본값은 0.2 (20%)입니다.

    Returns:
        tuple: (train_maps, val_maps) 형태의 튜플입니다.
               train_maps는 학습용 데이터 리스트, val_maps는 검증용 데이터 리스트입니다.
               입력 데이터가 없을 경우 빈 리스트들을 반환합니다.
    """
    # 만약 로드된 맵 데이터가 없다면 오류 메시지를 출력하고 빈 리스트 두 개를 반환합니다.
    if not all_maps: 
        print("오류: 로드된 맵 데이터가 없습니다. maps 폴더를 확인하세요.")
        return [], []
    
    random.shuffle(all_maps) # 데이터를 무작위로 섞어 편향을 줄입니다.

    # 검증용 데이터 크기를 계산합니다.
    # 데이터셋이 매우 작을 경우 (예: val_ratio에 따라 분리 시 한쪽이 0이 될 수 있는 경우)
    # 또는 데이터가 아예 없는 경우를 고려하여 val_size를 0으로 설정할 수 있도록 합니다.
    if len(all_maps) <= 1 / (1 - val_ratio) and len(all_maps) > 0 : # 데이터가 충분히 크지 않아 검증셋을 0으로 해야 하는 경우
        val_size = 0 
    elif not all_maps: # 데이터가 아예 없는 경우
        val_size = 0
    else: # 일반적인 경우, 전체 데이터 개수에 비율을 곱하여 검증 데이터 개수를 정합니다. 최소 1개를 보장하거나, 데이터가 1개면 0이 됩니다.
        val_size = max(1, int(len(all_maps) * val_ratio)) if len(all_maps) > 1 else 0

    # 계산된 검증용 데이터 크기에 따라 학습용과 검증용 데이터를 분리합니다.
    if val_size == 0 and len(all_maps) > 0 : 
        # 검증용 데이터가 없는 경우 (데이터가 매우 적거나 비율 설정상), 모든 데이터를 학습용으로 사용합니다.
        print("정보: 맵 데이터 수가 적거나 비율 설정으로 인해 검증용 데이터 없이 학습용으로만 사용합니다.")
        train_maps = all_maps
        val_maps = []
    elif not all_maps: # 데이터가 아예 없는 경우
        train_maps = []
        val_maps = []
    else: # 검증용 데이터가 있는 경우, 리스트 슬라이싱을 사용하여 분리합니다.
        val_maps = all_maps[:val_size]
        train_maps = all_maps[val_size:]
        
    return train_maps, val_maps # 분리된 학습용, 검증용 데이터 리스트를 반환합니다.

# 전체 데이터 준비 과정을 총괄하는 함수입니다.
# 맵 데이터를 로드하고, 학습용/검증용으로 분리한 후, 파일로 저장합니다.
def prepare_dataset(map_dir, val_ratio=0.2):
    """
    맵 데이터 준비의 전체 과정을 실행합니다.
    1. map_dir에서 맵 데이터를 로드합니다. (load_map_data 호출)
    2. 로드된 데이터를 학습용과 검증용으로 분리합니다. (split_train_val 호출)
    3. 분리된 데이터를 'data' 폴더 아래 'train_maps.json'과 'val_maps.json' 파일로 저장합니다.

    Args:
        map_dir (str): 맵 파일들이 저장된 디렉토리 경로입니다.
        val_ratio (float, optional): 전체 데이터 중 검증용으로 사용할 비율입니다. 기본값은 0.2입니다.

    Returns:
        tuple: (train_maps, val_maps) 형태의 튜플을 반환하거나,
               맵 파일 로드 실패 시 (None, None)을 반환합니다.
    """
    all_maps = load_map_data(map_dir) # 맵 데이터 로드를 시도합니다.
    # 맵 데이터 로드에 실패했거나 유효한 맵이 하나도 없는 경우, 메시지를 출력하고 None을 반환하여 중단합니다.
    if not all_maps:
        print("데이터 준비 중단: 맵 파일을 찾을 수 없거나 로드할 수 없습니다.")
        return None, None
        
    train_maps, val_maps = split_train_val(all_maps, val_ratio) # 로드된 데이터를 학습용과 검증용으로 분리합니다.
    
    # 처리된 맵의 총 개수, 학습용 및 검증용 맵의 개수를 출력합니다.
    print(f"총 맵 개수: {len(all_maps)}")
    print(f"학습용 맵 개수: {len(train_maps)}")
    print(f"검증용 맵 개수: {len(val_maps)}")
    
    os.makedirs("data", exist_ok=True) # 데이터를 저장할 'data' 디렉토리를 생성합니다. 이미 존재하면 오류를 발생시키지 않습니다.
    
    # 학습용 데이터가 있다면 'data/train_maps.json' 파일로 저장합니다.
    if train_maps:
        with open("data/train_maps.json", "w", encoding="utf-8") as f:
            # json.dump를 사용하여 리스트를 JSON 형식으로 파일에 씁니다.
            # ensure_ascii=False는 한글 등이 ASCII로 변환되지 않고 그대로 저장되도록 합니다.
            # indent=2는 JSON 파일을 사람이 읽기 쉽게 2칸 들여쓰기로 포맷팅합니다.
            json.dump(train_maps, f, ensure_ascii=False, indent=2)
    else:
        print("학습용 데이터가 없어 data/train_maps.json 파일을 생성하지 않습니다.")
        
    # 검증용 데이터가 있다면 'data/val_maps.json' 파일로 저장합니다.
    if val_maps:
        with open("data/val_maps.json", "w", encoding="utf-8") as f:
            json.dump(val_maps, f, ensure_ascii=False, indent=2)
    else:
        print("검증용 데이터가 없어 data/val_maps.json 파일을 생성하지 않습니다.")
        
    print("데이터셋이 준비되었습니다.")
    return train_maps, val_maps # 준비된 학습용, 검증용 데이터를 반환합니다.

# 이 스크립트가 직접 실행될 때만 아래 코드가 동작합니다.
# (다른 스크립트에서 이 파일을 import할 때는 실행되지 않습니다.)
if __name__ == "__main__":
    # 'maps' 디렉토리가 존재하지 않으면 생성하고 사용자에게 안내 메시지를 출력합니다.
    # 이 디렉토리는 원본 맵 데이터 파일(예: map_01.txt)을 저장하는 곳입니다.
    if not os.path.exists("maps"):
        os.makedirs("maps")
        print("'maps' 폴더가 생성되었습니다. 원본 맵 데이터 파일(예: map_01.txt)을 이 폴더에 넣어주세요.")
        
    map_dir = "maps"  # 맵 파일이 있는 디렉토리 경로를 설정합니다.
    val_ratio = 0.2  # 검증 데이터의 비율을 20%로 설정합니다.
    
    # 'maps' 디렉토리 안에 'map_*.txt' 형식의 파일이 하나도 없는지 확인합니다.
    if not any(Path(map_dir).glob("map_*.txt")):
        print(f"'{map_dir}' 폴더에 'map_*.txt' 형식의 파일이 없습니다. 맵 파일을 추가해주세요.")
    else:
        # 맵 파일이 있다면 데이터 준비 함수를 호출합니다.
        prepare_dataset(map_dir, val_ratio)
        print("1단계 완료: 데이터 준비가 끝났습니다.")