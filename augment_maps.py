import os # 운영 체제와 상호 작용하기 위한 모듈입니다. 파일 경로 생성, 디렉토리 확인 등에 사용됩니다.
from pathlib import Path # 파일 및 디렉토리 경로를 객체 지향적으로 다루기 위한 모듈입니다.

# 주어진 맵 라인들로부터 맵의 높이와 너비를 계산하는 함수입니다.
def get_map_dimensions(map_lines):
    """
    맵 데이터(문자열 리스트)를 입력받아 맵의 높이와 너비를 반환합니다.

    Args:
        map_lines (list): 맵의 각 줄을 문자열로 담고 있는 리스트입니다.

    Returns:
        tuple: (height, width) 형태의 튜플. 맵이 비어있으면 (0, 0)을 반환합니다.
    """
    # 맵 라인이 비어있으면 높이와 너비 모두 0을 반환합니다.
    if not map_lines:
        return 0, 0
    height = len(map_lines) # 높이는 라인 수와 같습니다.
    width = len(map_lines[0]) if height > 0 else 0 # 너비는 첫 번째 라인의 길이와 같습니다 (맵이 존재할 경우).
    # (선택 사항) 모든 라인이 동일한 너비인지 확인하는 코드를 추가할 수 있습니다.
    # 예: 모든 라인을 순회하며 너비가 일치하지 않으면 경고를 출력하거나 예외를 발생시킵니다.
    return height, width

# 맵 문자열을 좌우로 반전시키는 함수입니다.
def flip_map_horizontally(map_string):
    """
    주어진 맵 문자열을 좌우로 반전시킵니다.

    Args:
        map_string (str): 원본 맵 데이터 문자열입니다. 각 줄은 '\n'으로 구분됩니다.

    Returns:
        str: 좌우로 반전된 맵 데이터 문자열입니다.
    """
    map_lines = map_string.strip().split('\n') # 문자열 양 끝의 공백을 제거하고, 줄바꿈 문자를 기준으로 나눕니다.
    # 각 라인에 대해 문자열 슬라이싱(line[::-1])을 사용하여 좌우를 반전시킵니다.
    flipped_lines = [line[::-1] for line in map_lines]
    return "\n".join(flipped_lines) # 반전된 라인들을 다시 줄바꿈 문자로 합쳐 하나의 문자열로 만듭니다.

# 맵에서 'G' 문자의 두 칸 위 문자를 변경하는 함수입니다.
def change_char_above_g(map_string, original_char='-', new_char='W'):
    """
    맵 데이터에서 'G' 타일을 찾고, 그 타일의 두 칸 위에 있는 문자를 변경합니다.
    변경은 해당 위치의 문자가 'original_char'일 경우에만 'new_char'로 수행됩니다.
    첫 번째로 발견된 'G'에 대해서만 이 작업을 수행합니다.

    Args:
        map_string (str): 원본 맵 데이터 문자열입니다.
        original_char (str, optional): 변경 대상 위치의 원래 문자여야 하는 값. 기본값은 '-'입니다.
        new_char (str, optional): 변경할 새로운 문자. 기본값은 'W'입니다.

    Returns:
        str: 문자가 성공적으로 변경된 경우 수정된 맵 문자열을 반환합니다.
             'G'를 찾지 못했거나, 대상 위치가 유효하지 않거나,
             'original_char'와 일치하지 않아 변경이 일어나지 않은 경우에는 원본 맵 문자열을 반환합니다.
    """
    map_lines = map_string.strip().split('\n') # 맵 문자열을 라인 리스트로 변환합니다.
    height, width = get_map_dimensions(map_lines) # 맵의 크기를 가져옵니다.
    
    # 문자열은 불변(immutable) 객체이므로, 직접 수정할 수 없습니다.
    # 따라서 각 라인을 문자 리스트로 변환하여 수정이 가능하도록 합니다.
    new_map_lines_list = [list(line) for line in map_lines]

    g_found_and_changed = False # 'G'를 찾고 성공적으로 문자를 변경했는지 여부를 나타내는 플래그입니다.
    for r in range(height): # 맵의 모든 행(row)을 순회합니다.
        for c in range(width): # 맵의 모든 열(column)을 순회합니다.
            if map_lines[r][c] == 'G': # 'G' 문자를 찾으면
                target_row = r - 2 # 'G'의 두 칸 위 행 위치를 계산합니다.
                target_col = c     # 'G'와 동일한 열 위치입니다.
                
                # 대상 위치가 맵 범위 내에 있고, 해당 위치의 문자가 'original_char'와 일치하는지 확인합니다.
                if 0 <= target_row < height and \
                   0 <= target_col < width and \
                   new_map_lines_list[target_row][target_col] == original_char:
                    
                    new_map_lines_list[target_row][target_col] = new_char # 문자를 변경합니다.
                    g_found_and_changed = True # 변경 성공 플래그를 설정합니다.
                    # 일반적으로 맵에 'G'는 하나만 있다고 가정하고, 첫 번째 'G'에 대해서만 처리 후 루프를 종료합니다.
                    # 여러 'G'에 대해 모두 처리하려면 아래 break 문들을 제거해야 합니다.
                    break 
            if g_found_and_changed: # 내부 루프에서 변경이 발생했으면 외부 루프도 종료합니다.
                break
        if g_found_and_changed: # 외부 루프에서 변경이 발생했으면 더 이상 순회하지 않고 종료합니다.
            break
            
    if g_found_and_changed: # 변경이 성공적으로 이루어졌으면,
        # 수정된 문자 리스트들을 다시 문자열로 합쳐서 반환합니다.
        return "\n".join("".join(row_list) for row_list in new_map_lines_list)
    else:
        # 'G'를 찾지 못했거나 조건에 맞지 않아 변경이 일어나지 않은 경우, 원본 맵 문자열을 그대로 반환합니다.
        return map_string 

# 아래는 위에 정의된 함수들을 사용하여 맵 데이터를 증강하는 예시입니다.
def augment_map_data(original_map_data_str):
    """
    주어진 원본 맵 데이터 문자열에 대해 여러 가지 증강 기법을 적용합니다.

    Args:
        original_map_data_str (str): 원본 맵 데이터 문자열입니다.

    Returns:
        list: 각 요소는 증강된 맵 데이터와 증강 타입을 담은 딕셔셔니입니다.
              예: [{"map_str": "...", "type": "flipped"}, {"map_str": "...", "type": "g_above_w_orig"}]
    """
    augmented_maps = [] # 증강된 맵들을 저장할 리스트입니다.

    # 1. 좌우 반전 증강
    flipped_map_str = flip_map_horizontally(original_map_data_str)
    augmented_maps.append({"map_str": flipped_map_str, "type": "flipped"})

    # 2. 원본 맵 기준: 'G' 두 칸 위 '-'를 'W'로 변경
    g_changed_map_str_orig = change_char_above_g(original_map_data_str, original_char='-', new_char='W')
    # 변경이 실제로 일어났을 경우에만 증강된 맵 리스트에 추가합니다.
    if g_changed_map_str_orig != original_map_data_str: 
        augmented_maps.append({"map_str": g_changed_map_str_orig, "type": "g_above_w_orig"})

    # 3. 좌우 반전된 맵 기준: 'G' 두 칸 위 '-'를 'W'로 변경
    # 좌우 반전으로 인해 'G'의 위치가 변경되므로, 이 증강은 새로운 결과를 생성할 수 있습니다.
    g_changed_map_str_flipped = change_char_above_g(flipped_map_str, original_char='-', new_char='W')
    if g_changed_map_str_flipped != flipped_map_str: # 변경이 일어났을 경우에만 추가
        augmented_maps.append({"map_str": g_changed_map_str_flipped, "type": "g_above_w_flipped"})
            
    return augmented_maps


# 이 스크립트가 직접 실행될 때만 아래 코드가 동작합니다.
if __name__ == '__main__':
    # 이 스크립트를 직접 실행하면, 'maps' 폴더에 있는 원본 맵 파일들을 읽어와
    # 데이터 증강을 수행하고, 그 결과를 'augmented_maps' 폴더에 저장하는 예시입니다.
    
    source_map_dir = "maps"  # 원본 맵 파일이 있는 디렉토리입니다.
    augmented_map_dir = "augmented_maps" # 증강된 맵 파일을 저장할 디렉토리입니다.
    
    os.makedirs(augmented_map_dir, exist_ok=True) # 저장할 디렉토리가 없으면 생성합니다.

    # 원본 맵 디렉토리에서 'map_*.txt' 패턴의 파일 목록을 가져옵니다.
    map_files = list(Path(source_map_dir).glob("map_*.txt"))
    if not map_files:
        print(f"'{source_map_dir}' 폴더에 'map_*.txt' 파일이 없습니다.")
    else:
        print(f"총 {len(map_files)}개의 원본 맵 파일을 찾았습니다.")

    total_augmented_count = 0 # 총 생성된 증강 맵 파일 수를 기록합니다.
    for map_file_path in map_files: # 각 원본 맵 파일에 대해 반복합니다.
        with open(map_file_path, "r", encoding="utf-8") as f: # 파일을 읽기 모드로 엽니다.
            content = f.read() # 파일 전체 내용을 읽습니다.
            
            prompt = "" # 프롬프트를 저장할 변수입니다.
            map_data_original = "" # 원본 맵 데이터를 저장할 변수입니다.

            # 파일 내용에서 프롬프트와 맵 데이터를 구분하는 태그들을 정의합니다.
            prompt_start_tag = "<PROMPT>"
            prompt_end_tag = "</PROMPT>"
            map_start_tag = "<MAP>"
            map_end_tag = "</MAP>"

            # 프롬프트 부분을 추출합니다.
            idx_prompt_start = content.find(prompt_start_tag)
            idx_prompt_end = content.find(prompt_end_tag)
            if idx_prompt_start != -1 and idx_prompt_end != -1 and idx_prompt_start < idx_prompt_end:
                prompt = content[idx_prompt_start + len(prompt_start_tag):idx_prompt_end].strip()
            
            # 맵 데이터 부분을 추출합니다.
            idx_map_start = content.find(map_start_tag)
            idx_map_end = content.find(map_end_tag)
            if idx_map_start != -1 and idx_map_end != -1 and idx_map_start < idx_map_end:
                map_data_original = content[idx_map_start + len(map_start_tag):idx_map_end].strip()
            else: # 태그 형식이 올바르지 않으면 경고를 출력하고 다음 파일로 넘어갑니다.
                print(f"경고: {map_file_path.name} 파일의 태그 형식이 올바르지 않거나 맵 데이터를 찾을 수 없습니다. 건너뜁니다.")
                continue
            
            # 맵 데이터가 비어있으면 경고를 출력하고 다음 파일로 넘어갑니다.
            if not map_data_original.strip():
                print(f"경고: {map_file_path.name} 파일에서 맵 데이터를 추출하지 못했습니다. 건너뜁니다.")
                continue

            # 원본 맵 데이터에 대해 증강 함수를 호출합니다.
            augmented_results = augment_map_data(map_data_original)
            
            print(f"\n'{map_file_path.name}'에 대한 증강 결과:")
            for aug_map_info in augmented_results: # 각 증강 결과에 대해 반복합니다.
                map_str = aug_map_info["map_str"] # 증강된 맵 문자열입니다.
                aug_type = aug_map_info["type"]   # 증강 타입입니다 (예: "flipped").
                
                # 저장할 새로운 파일 이름을 생성합니다. (예: map_01_flipped.txt)
                new_filename = Path(augmented_map_dir) / f"{map_file_path.stem}_{aug_type}{map_file_path.suffix}"
                
                # 증강된 맵 데이터를 새 파일에 저장합니다.
                with open(new_filename, "w", encoding="utf-8") as nf:
                    # 프롬프트는 원본을 사용하거나, 증강 타입을 명시하는 내용을 추가할 수 있습니다.
                    nf.write(f"<PROMPT>{prompt} (aug: {aug_type})</PROMPT>\n<MAP>\n{map_str}\n</MAP>")
                print(f"  저장: {new_filename.name}")
                total_augmented_count +=1 # 증강 파일 카운트를 증가시킵니다.

    print(f"\n총 {total_augmented_count}개의 증강된 맵 파일을 '{augmented_map_dir}' 폴더에 저장했습니다.")
    print("이제 'maps' 폴더 대신 'augmented_maps' 폴더 (또는 두 폴더를 합쳐서) 1단계 데이터 준비에 사용할 수 있습니다.")
    print("또는, 이 파일의 함수들을 1_data_preparation.py 스크립트에 통합하여 데이터 로드 시점에 증강을 적용할 수도 있습니다.")