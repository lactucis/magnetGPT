import os # 운영 체제와 상호 작용하기 위한 모듈 (파일 경로, 디렉토리 생성 등)
import torch # PyTorch 라이브러리 (텐서 연산, 모델 로드 및 실행)
from transformers import GPT2Tokenizer, GPT2LMHeadModel # Hugging Face Transformers 라이브러리 (GPT-2 토크나이저 및 모델)
import argparse # 명령줄 인자 파싱을 위한 모듈
from datetime import datetime # 현재 시간을 가져오기 위한 모듈 (파일 이름 생성 시 사용)
import random # 무작위 선택 및 셔플을 위한 모듈

# ----- 생성될 맵의 실제 크기 및 주요 게임 규칙 정의 -----
# 이 값들은 생성된 맵의 후처리 과정에서 사용되어 맵의 일관성과 플레이 가능성을 보장합니다.
MAP_WIDTH = 40  # 생성될 맵의 가로 타일 수 (실제 게임 맵 크기에 맞게 조정 필요)
MAP_HEIGHT = 25 # 생성될 맵의 세로 타일 수 (실제 게임 맵 크기에 맞게 조정 필요)

# P(플레이어 시작점)와 G(목표 지점) 사이의 최소 맨해튼 거리 (대각선 이동 없이 가로, 세로로만 이동했을 때의 거리)
MIN_DISTANCE_PG = max(5, int(min(MAP_WIDTH, MAP_HEIGHT) * 0.4)) # 맵 크기의 40% 또는 최소 5칸 (기존 0.3에서 증가, 1.5는 너무 클 수 있음)
# O(특수 오브젝트)가 P 및 G로부터 떨어져야 하는 최소 맨해튼 거리
MIN_DISTANCE_O_FROM_PG = max(3, int(MIN_DISTANCE_PG * 0.5)) # P-G 거리의 절반 또는 최소 3칸 (기존 P-G 거리의 1.5배는 너무 클 수 있음)

# 빈 공간에 무작위로 추가할 W(벽 또는 플랫폼) 플랫폼의 최대 개수 (0으로 설정 시 추가 안 함)
MAX_RANDOM_W_PLATFORMS = 5 # 기존 10개에서 줄임 (너무 많으면 맵이 복잡해질 수 있음, 결과 보면서 조절)
# 무작위 W 플랫폼의 길이 (예: 1이면 'W', 2이면 'WW', 3이면 'WWW')
RANDOM_W_PLATFORM_LENGTH = 2 # 'WWW'는 때로 너무 길 수 있어 'WW'로 조정, 결과 보면서 조절
# -------------------------------------------------

# 학습된 모델과 토크나이저를 로드하는 함수입니다.
def load_trained_model(model_dir="trained_model"):
    """
    지정된 디렉토리에서 학습 완료된 모델과 토크나이저를 로드합니다.

    Args:
        model_dir (str, optional): 모델과 토크나이저가 저장된 디렉토리 경로. 기본값은 "trained_model"입니다.

    Returns:
        tuple: (model, tokenizer) 튜플. 로드 실패 시 (None, None)을 반환합니다.
    """
    try:
        # 지정된 디렉토리에서 토크나이저를 로드합니다.
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        # 지정된 디렉토리에서 모델을 로드합니다.
        model = GPT2LMHeadModel.from_pretrained(model_dir)
        print(f"'{model_dir}'에서 모델과 토크나이저를 성공적으로 로드했습니다.")
        return model, tokenizer
    except Exception as e:
        # 로드 중 오류 발생 시 메시지를 출력하고 None을 반환합니다.
        print(f"모델 로드 중 오류 발생: {e}")
        print(f"'{model_dir}' 폴더에 학습된 모델과 토크나이저가 있는지 확인해주세요. (3_train_model.py 실행 필요)")
        return None, None

# 모델이 학습한 특수 토큰 표현(예: '[P]')을 원래의 단일 문자(예: 'P')로 되돌리는 함수입니다.
def revert_map_representation(mangled_map_text):
    """
    모델 생성 결과물에 포함된 특수 토큰 표현 ('[P]', '[W]' 등)을
    원래의 단일 문자 ('P', 'W' 등)로 변환합니다.

    Args:
        mangled_map_text (str): 모델이 생성한, 특수 토큰 표현을 포함하는 맵 텍스트입니다.

    Returns:
        str: 원래의 문자 표현으로 변환된 맵 텍스트입니다.
    """
    # 변환 규칙을 정의한 딕셔너리입니다.
    revert_tile_map = {
        "[P]": "P", "[W]": "W", "[-]": "-",
        "[T]": "T", "[G]": "G", "[O]": "O",
    }
    processed_map_text = mangled_map_text
    # 정의된 규칙에 따라 모든 특수 토큰 표현을 원래 문자로 치환합니다.
    for new_rep, original_char in revert_tile_map.items():
        processed_map_text = processed_map_text.replace(new_rep, original_char)
    return processed_map_text

# 'W' 타일의 연속성을 강화하기 위한 간단한 후처리 함수입니다 (선택적 사용).
def apply_w_continuity(map_list, map_width, map_height):
    """
    맵 리스트에서 'W' 타일의 연속성을 강화하려고 시도합니다.
    예를 들어, '-W-' 패턴을 'WWW'로 변경하거나 'WW-'를 'WWW'로 확장합니다.
    이 함수는 map_list를 직접 수정합니다.

    Args:
        map_list (list): 2차원 리스트 형태의 맵 데이터 (각 내부 리스트는 맵의 한 행을 나타내는 문자 리스트).
        map_width (int): 맵의 너비.
        map_height (int): 맵의 높이.
    """
    # '-W-' 패턴을 'WWW'로 변경 (가운데 W를 기준으로 양 옆이 '-'이면 W로 채움)
    for r_idx in range(map_height):
        for c_idx in range(1, map_width - 1): # 맵 가장자리는 제외
            if map_list[r_idx][c_idx] == 'W' and \
               map_list[r_idx][c_idx-1] == '-' and \
               map_list[r_idx][c_idx+1] == '-':
                # 추가 조건 (예: P, G, O 바로 옆은 아닌지 등)을 고려할 수 있습니다.
                map_list[r_idx][c_idx-1] = 'W'
                map_list[r_idx][c_idx+1] = 'W'
    
    # 'WW-' 패턴을 'WWW'로, '-WW' 패턴을 'WWW'로 확장 (더 많은 W 생성 가능성)
    for r_idx in range(map_height):
        for c_idx in range(map_width):
            if map_list[r_idx][c_idx] == 'W':
                # 오른쪽으로 확장: 현재 위치와 다음 위치가 'W'이고, 그 다음이 '-'이면 'W'로 채움
                if c_idx + 2 < map_width and map_list[r_idx][c_idx+1] == 'W' and map_list[r_idx][c_idx+2] == '-':
                    map_list[r_idx][c_idx+2] = 'W'
                # 왼쪽으로 확장: 현재 위치와 이전 위치가 'W'이고, 그 이전이 '-'이면 'W'로 채움
                if c_idx - 2 >= 0 and map_list[r_idx][c_idx-1] == 'W' and map_list[r_idx][c_idx-2] == '-':
                    map_list[r_idx][c_idx-2] = 'W'

# 맵 데이터(2차원 리스트)에 P, G, O 관련 규칙, 착지 플랫폼, T 규칙, 빈 공간 W 플랫폼 추가, W 연속성, 테두리 규칙 등을 적용하는 함수입니다.
def ensure_rules_on_map_list(map_list, map_width, map_height, prompt_str=""):
    """
    생성된 맵(2차원 리스트)에 게임 규칙을 적용하여 맵의 유효성을 보장합니다.
    이 함수는 입력된 map_list를 직접 수정합니다.

    주요 규칙:
    1. P, G, O는 각각 정확히 1개만 존재하도록 합니다.
    2. P, G, O 아래에는 착지 플랫폼('W')을 배치합니다.
    3. P와 G, O와 P/G 간의 최소 거리를 유지하려고 시도합니다.
    4. 프롬프트에 "no traps"가 없으면 T(함정)는 'W'에 연결되도록 합니다.
    5. 빈 공간에 무작위로 작은 'W' 플랫폼을 추가합니다.
    6. (선택적) 'W' 타일의 연속성을 강화합니다.
    7. 맵의 테두리를 'W'로 채웁니다 (P, G, O는 보호).

    Args:
        map_list (list): 2차원 리스트 형태의 맵 데이터.
        map_width (int): 맵의 너비.
        map_height (int): 맵의 높이.
        prompt_str (str, optional): 맵 생성 시 사용된 프롬프트 문자열 (T 규칙 적용에 사용).
    """
    p_locations, g_locations, o_locations, t_locations = [], [], [], []
    
    # 현재 맵에서 P,G,O,T 및 빈칸(-)의 위치를 스캔하는 내부 함수입니다.
    def scan_map_elements():
        p_locs, g_locs, o_locs, t_locs, empty_locs = [], [], [], [], []
        for r_idx, row_list_chars in enumerate(map_list):
            for c_idx, tile in enumerate(row_list_chars):
                if tile == 'P': p_locs.append((r_idx, c_idx))
                elif tile == 'G': g_locs.append((r_idx, c_idx))
                elif tile == 'O': o_locs.append((r_idx, c_idx))
                elif tile == 'T': t_locs.append((r_idx, c_idx))
                elif tile == '-': empty_locs.append((r_idx,c_idx))
        return p_locs, g_locs, o_locs, t_locs, empty_locs

    p_locations, g_locations, o_locations, t_locations, empty_locations = scan_map_elements()
    random.shuffle(empty_locations) # 빈칸 위치를 무작위로 섞습니다 (랜덤 배치 시 사용).

    # --- P(플레이어 시작점) 처리: 정확히 1개 존재하도록 하고, 바로 아래에 'W' 착지 플랫폼을 놓습니다. ---
    if len(p_locations) > 1: # P가 여러 개 있으면 첫 번째 P만 남기고 나머지는 '-'로 변경합니다.
        for r_p_old, c_p_old in p_locations[1:]: map_list[r_p_old][c_p_old] = '-'
        p_locations = [p_locations[0]]
    elif not p_locations: # P가 없으면 새로 배치합니다.
        placed_p = False
        # 우선 맵 하단 중앙 근처의 빈칸에 배치 시도합니다.
        p_r_cand, p_c_cand = max(0, map_height - 2), map_width // 2 # P는 너무 아래에 있지 않도록 (착지 플랫폼 공간)
        if (p_r_cand + 1 < map_height) and (map_list[p_r_cand][p_c_cand] == '-'):
            map_list[p_r_cand][p_c_cand] = 'P'
            p_locations = [(p_r_cand, p_c_cand)]
            placed_p = True
        
        if not placed_p and empty_locations: # 그래도 배치되지 않았고 빈칸이 있다면, 첫 번째 유효한 빈칸에 배치합니다.
            for r_p, c_p in empty_locations:
                if r_p + 1 < map_height and map_list[r_p][c_p] == '-': # 착지 플랫폼을 놓을 수 있는 위치
                    map_list[r_p][c_p] = 'P'
                    p_locations = [(r_p, c_p)]
                    placed_p = True; break
        if not placed_p: # 최후의 수단으로 강제 배치 (맵 하단 중앙, 겹치면 옆으로)
            p_r_f, p_c_f = max(0, map_height - 2), map_width // 2
            map_list[p_r_f][p_c_f] = 'P'
            p_locations = [(p_r_f, p_c_f)]
            print("경고: P를 찾을 수 없어 강제로 배치했습니다.")
    # P 착지 플랫폼('W') 추가
    if p_locations:
        pr, pc = p_locations[0]
        if pr + 1 < map_height and map_list[pr+1][pc] not in ['G', 'O']: # G, O가 있는 자리는 피함
            map_list[pr+1][pc] = 'W'

    # P 배치 후 빈칸 목록 업데이트
    _, g_locations, o_locations, t_locations, empty_locations = scan_map_elements() # P는 확정, G,O,T,빈칸 재스캔
    random.shuffle(empty_locations)

    # --- G(목표 지점) 처리: 정확히 1개, 'W' 위에, P와 일정 거리 이상 유지 시도 ---
    g_candidates_far, g_candidates_near = [], [] # P로부터 먼 후보, 가까운 후보
    if p_locations: # P가 있어야 거리를 계산할 수 있습니다.
        pr_p, pc_p = p_locations[0]
        for r_cand, c_cand in empty_locations: # 현재 빈칸 중에서 G 후보를 찾습니다.
            if r_cand + 1 < map_height and map_list[r_cand][c_cand] == '-': # 착지 플랫폼 공간이 있고, 현재 위치가 빈칸
                distance_to_p = abs(r_cand - pr_p) + abs(c_cand - pc_p) # 맨해튼 거리 계산
                if distance_to_p >= MIN_DISTANCE_PG:
                    g_candidates_far.append((r_cand, c_cand))
                else:
                    g_candidates_near.append((r_cand, c_cand))
    else: # P가 없는 (이론상 발생하면 안되는) 경우, 모든 유효 빈칸을 후보로 간주합니다.
        g_candidates_far = [loc for loc in empty_locations if loc[0]+1 < map_height and map_list[loc[0]][loc[1]] == '-']
            
    final_g_candidates = g_candidates_far + g_candidates_near # 먼 후보 우선, 없으면 가까운 후보라도 사용

    if len(g_locations) > 1: # G가 여러 개면 첫 번째 G만 남깁니다.
        for r_g_old, c_g_old in g_locations[1:]: map_list[r_g_old][c_g_old] = '-'
        g_locations = [g_locations[0]]
    elif not g_locations: # G가 없으면 새로 배치합니다.
        placed_g = False
        if final_g_candidates: # 후보 위치가 있다면
            for r_g, c_g in final_g_candidates:
                if map_list[r_g][c_g] == '-': # 여전히 빈칸인지 확인 (P가 차지했을 수 있음)
                    map_list[r_g][c_g] = 'G'
                    g_locations = [(r_g,c_g)]
                    placed_g = True; break
        if not placed_g: # 후보가 없거나 모두 사용 불가 시 강제 배치 (맵 상단 P 반대편 근처)
            g_r_f, g_c_f = min(1, map_height - 2), (map_width // 2 + int(map_width*0.2) + map_width)%map_width 
            if p_locations and p_locations[0] == (g_r_f, g_c_f): g_c_f = (g_c_f + 1) % map_width # P와 겹치면 한 칸 옆으로
            map_list[g_r_f][g_c_f] = 'G'
            g_locations = [(g_r_f, g_c_f)]
            print("경고: G를 배치할 적절한 위치를 찾지 못해 강제로 배치했습니다.")
    # G 착지 플랫폼('W') 추가
    if g_locations:
        gr, gc = g_locations[0]
        if gr + 1 < map_height and map_list[gr+1][gc] not in ['P', 'O']: # P, O가 있는 자리는 피함
            map_list[gr+1][gc] = 'W'

    # G 배치 후 빈칸 목록 업데이트
    _, _, o_locations, t_locations, empty_locations = scan_map_elements() # P,G 확정. O,T,빈칸 재스캔
    random.shuffle(empty_locations)

    # --- O(특수 오브젝트) 처리: 정확히 1개, 'W' 위에, P/G와 일정 거리 이상 유지 시도 ---
    o_candidates_far, o_candidates_near = [], [] # P/G로부터 먼 후보, 가까운 후보
    for r_cand, c_cand in empty_locations:
        if r_cand + 1 < map_height and map_list[r_cand][c_cand] == '-': # 착지 공간 있고 빈칸
            dist_to_p = float('inf')
            dist_to_g = float('inf')
            if p_locations: dist_to_p = abs(r_cand - p_locations[0][0]) + abs(c_cand - p_locations[0][1])
            if g_locations: dist_to_g = abs(r_cand - g_locations[0][0]) + abs(c_cand - g_locations[0][1])
            
            if dist_to_p >= MIN_DISTANCE_O_FROM_PG and dist_to_g >= MIN_DISTANCE_O_FROM_PG:
                o_candidates_far.append((r_cand, c_cand))
            else:
                o_candidates_near.append((r_cand, c_cand))
    final_o_candidates = o_candidates_far + o_candidates_near

    if len(o_locations) > 1: # O가 여러 개면 첫 번째 O만 남깁니다.
        for r_o_old, c_o_old in o_locations[1:]: map_list[r_o_old][c_o_old] = '-'
        o_locations = [o_locations[0]]
    elif not o_locations: # O가 없으면 새로 배치합니다.
        placed_o = False
        if final_o_candidates:
            for r_o, c_o in final_o_candidates:
                if map_list[r_o][c_o] == '-': # 여전히 빈칸인지 확인
                    map_list[r_o][c_o] = 'O'
                    o_locations = [(r_o,c_o)]
                    placed_o = True; break
        if not placed_o: # 강제 배치 (맵 중앙 P/G 피해서)
            o_r_f, o_c_f = map_height // 2, (map_width // 2 - int(map_width*0.2) + map_width) % map_width
            # P 또는 G와 겹치지 않도록 위치 조정
            while (o_r_f, o_c_f) in p_locations or (o_r_f, o_c_f) in g_locations:
                o_c_f = (o_c_f - 1 + map_width) % map_width # 한 칸씩 옆으로 이동
                # 한 바퀴 돌았으면 행도 변경
                if o_c_f == ((map_width // 2 - int(map_width*0.2) + map_width) % map_width) : 
                    o_r_f = (o_r_f -1 + map_height)%map_height
            map_list[o_r_f][o_c_f] = 'O'
            o_locations = [(o_r_f, o_c_f)]
            print("경고: O를 배치할 적절한 위치를 찾지 못해 강제로 배치했습니다.")
    # O 착지 플랫폼('W') 추가
    if o_locations:
        or_o, oc_o = o_locations[0]
        if or_o + 1 < map_height and map_list[or_o+1][oc_o] not in ['P', 'G']: # P, G가 있는 자리는 피함
            map_list[or_o+1][oc_o] = 'W'

    # --- T(함정) 규칙 적용 (프롬프트 기반) ---
    # "no traps" 프롬프트가 있으면 T를 모두 '-'로 변경, 그렇지 않으면 T가 'W'에 연결되어 있는지 확인
    create_traps = "no traps" not in prompt_str.lower()
    # P,G,O 배치로 인해 T가 덮어씌워졌을 수 있으므로 T 위치 재스캔
    current_t_locations = []
    for r_idx, row_list_chars in enumerate(map_list):
        for c_idx, tile in enumerate(row_list_chars):
            if tile == 'T': current_t_locations.append((r_idx, c_idx))

    for r_t, c_t in current_t_locations: # 현재 맵에 있는 T들에 대해
        if not create_traps: # "no traps" 프롬프트가 있으면
            map_list[r_t][c_t] = '-' # T를 빈칸으로 변경
            continue
        # "no traps"가 아니면, T가 W에 연결되어 있는지 확인
        is_connected_to_w = False
        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]: # 상하좌우 인접 칸 확인
            nr, nc = r_t + dr, c_t + dc
            if 0 <= nr < map_height and 0 <= nc < map_width and map_list[nr][nc] == 'W':
                is_connected_to_w = True; break
        if not is_connected_to_w: # W에 연결되지 않은 T는 빈칸으로 변경
            map_list[r_t][c_t] = '-'


    # --- 빈 공간에 작은 W 플랫폼 몇 개 추가 ---
    # P,G,O,T 배치 후 남은 최종 빈칸 목록을 다시 스캔합니다.
    final_empty_locations = []
    for r_idx, row_list_chars in enumerate(map_list):
        for c_idx, tile in enumerate(row_list_chars):
            if tile == '-': final_empty_locations.append((r_idx,c_idx))
    random.shuffle(final_empty_locations)

    w_platforms_added = 0
    # 추가할 W 플랫폼을 놓을 충분한 빈 공간이 있는지 확인
    if len(final_empty_locations) > (MAX_RANDOM_W_PLATFORMS * RANDOM_W_PLATFORM_LENGTH) :
        for _ in range(MAX_RANDOM_W_PLATFORMS): # 최대 개수만큼 시도
            if not final_empty_locations: break # 더 이상 빈칸이 없으면 중단
            
            r_w_start, c_w_start = -1, -1 # 플랫폼 시작 위치 초기화
            # 플랫폼을 놓을 안전한 시작 위치를 찾습니다.
            for r_cand_w, c_cand_w in final_empty_locations:
                is_safe_to_place = True
                # 플랫폼 길이만큼 타일들을 놓을 수 있는지, P/G/O나 그 착지점을 덮어쓰지 않는지 확인
                for i in range(RANDOM_W_PLATFORM_LENGTH):
                    if not (c_cand_w + i < map_width): is_safe_to_place = False; break # 맵 너비 초과
                    # P,G,O 또는 그 바로 아래(착지 W 예상 지점)와 겹치는지 확인
                    for pr_major, pc_major in p_locations + g_locations + o_locations: # 주요 객체 위치들
                        if (r_cand_w == pr_major and c_cand_w + i == pc_major) or \
                           (r_cand_w == pr_major + 1 and c_cand_w + i == pc_major) : # 객체 또는 객체 바로 아래
                            is_safe_to_place = False; break
                    if not is_safe_to_place: break
                if is_safe_to_place: # 안전한 위치를 찾으면
                    r_w_start, c_w_start = r_cand_w, c_cand_w
                    break # 위치 탐색 중단
            
            if r_w_start != -1: # 유효한 시작 위치를 찾았으면
                can_place_platform = True
                temp_used_for_platform = [] # 실제로 플랫폼이 놓일 위치들
                # 플랫폼 길이만큼 '-'인지 다시 한번 확인 (다른 플랫폼과 겹치지 않도록)
                for i in range(RANDOM_W_PLATFORM_LENGTH):
                    if not (c_w_start + i < map_width and map_list[r_w_start][c_w_start+i] == '-'):
                        can_place_platform = False; break
                    temp_used_for_platform.append((r_w_start, c_w_start+i))
                
                if can_place_platform: # 플랫폼을 놓을 수 있으면
                    for r_plat, c_plat in temp_used_for_platform:
                        map_list[r_plat][c_plat] = 'W' # 'W'로 변경
                        if (r_plat, c_plat) in final_empty_locations: 
                            final_empty_locations.remove((r_plat, c_plat)) # 사용된 빈칸은 목록에서 제거
                    w_platforms_added += 1
            if w_platforms_added >= MAX_RANDOM_W_PLATFORMS: break # 최대 개수만큼 추가했으면 중단

    # --- W 연속성 강화 시도 (선택적 기능, 필요시 주석 해제) ---
    # apply_w_continuity(map_list, map_width, map_height)

    # --- 맵 테두리를 'W'로 만들기 (단, P, G, O는 보호) ---
    for r_idx in range(map_height):
        for c_idx in range(map_width):
            # 현재 위치가 첫 행/마지막 행 또는 첫 열/마지막 열인지 확인 (테두리인지)
            if r_idx == 0 or r_idx == map_height - 1 or \
               c_idx == 0 or c_idx == map_width - 1:
                # 해당 위치의 타일이 P, G, O가 아니면 'W'로 변경
                if map_list[r_idx][c_idx] not in ['P', 'G', 'O']:
                    map_list[r_idx][c_idx] = 'W'
    
    # 최종적으로 P, G, O 개수가 정확히 하나씩인지 확인 (디버깅용)
    final_p_count = sum(row.count('P') for row in map_list)
    final_g_count = sum(row.count('G') for row in map_list)
    final_o_count = sum(row.count('O') for row in map_list)
    if final_p_count != 1 or final_g_count != 1 or final_o_count != 1:
        print(f"경고: 최종 규칙 적용 후 P({final_p_count}), G({final_g_count}), O({final_o_count}) 개수가 정확히 하나씩이 아닐 수 있습니다. 이는 로직 오류 또는 충돌 가능성을 의미합니다.")


# 모델이 생성한 원본 텍스트를 필터링하고, 지정된 맵 크기에 맞게 재구성한 후, 게임 규칙을 적용하는 함수입니다.
def filter_and_reconstruct_map(raw_reverted_text, map_width, map_height, prompt_str=""):
    """
    모델이 생성하고 원래 문자 표현으로 되돌린 맵 텍스트에서 유효한 타일만 필터링하고,
    정의된 맵 크기(map_width, map_height)에 맞게 재구성한 후,
    ensure_rules_on_map_list 함수를 호출하여 최종 게임 규칙을 적용합니다.

    Args:
        raw_reverted_text (str): revert_map_representation을 거친 맵 텍스트.
        map_width (int): 목표 맵 너비.
        map_height (int): 목표 맵 높이.
        prompt_str (str, optional): 맵 생성 시 사용된 프롬프트 (규칙 적용 시 필요).

    Returns:
        str: 필터링, 재구성 및 규칙 적용이 완료된 최종 맵 텍스트.
    """
    # 유효한 원본 타일 문자 집합 (줄바꿈 포함)
    valid_tiles_original = {'P', 'W', '-', 'G', 'O', 'T', '\n'}
    filtered_chars = [] # 유효한 문자만 저장할 리스트
    
    # 생성된 텍스트를 줄 단위로 나누고, 각 문자가 유효한 타일인지 확인하여 필터링합니다.
    # 원본 줄바꿈도 유지하려고 시도합니다.
    for char_or_substring in raw_reverted_text.split('\n'): # 먼저 줄바꿈으로 나눔
        for char in char_or_substring: # 각 줄 내의 문자들을 순회
            if char in valid_tiles_original: # 유효한 타일 문자('P', 'W', '-', 'G', 'O', 'T')만 추가
                filtered_chars.append(char)
        if char_or_substring: # 빈 줄이 아니었다면, 줄바꿈 문자를 추가 (원래 줄 구조 유지 시도)
            filtered_chars.append('\n')
    if filtered_chars and filtered_chars[-1] == '\n': # 마지막에 불필요한 줄바꿈이 있으면 제거
        filtered_chars.pop()

    final_map_list = [] # 최종 맵을 2차원 리스트 형태로 저장할 변수
    
    # 필터링 후 유효 타일 수가 매우 적으면 (맵 면적의 5% 또는 10개 미만),
    # 기본 빈 맵('-'으로 채워진 맵)에서 시작하여 규칙을 적용합니다.
    # (P,G,O 등은 ensure_rules_on_map_list 함수 내에서 강제 배치될 수 있음)
    if len(filtered_chars) < min(map_width * map_height * 0.05, 10) :
        print("경고: 모델 생성 결과에서 유효한 맵 타일이 매우 적습니다. 기본 빈 맵 구조에 규칙을 적용합니다.")
        final_map_list = [list('-' * map_width) for _ in range(map_height)]
    else:
        # 유효 타일이 충분하면, 이를 사용하여 맵을 재구성합니다.
        reconstructed_map_lines = [] # 재구성된 맵의 각 줄을 저장할 리스트
        char_iterator = iter(filtered_chars) # 필터링된 문자들을 순차적으로 가져오기 위한 이터레이터
        try:
            for r in range(map_height): # 목표 맵 높이만큼 반복
                current_line_chars = [] # 현재 만들고 있는 줄의 문자들
                for c in range(map_width): # 목표 맵 너비만큼 반복
                    char = next(char_iterator, '-') # 이터레이터에서 다음 문자를 가져옴. 문자가 없으면 '-'로 채움.
                    if char == '\n': # 줄바꿈 문자를 만나면
                        # 현재 줄의 남은 부분을 '-'로 채우고 다음 줄로 넘어감.
                        current_line_chars.extend(['-'] * (map_width - len(current_line_chars)))
                        break 
                    current_line_chars.append(char) # 현재 문자를 줄에 추가
                # 한 줄 완성 후, 혹시 너비가 모자라면 '-'로 채웁니다.
                if len(current_line_chars) < map_width:
                    current_line_chars.extend(['-'] * (map_width - len(current_line_chars)))
                reconstructed_map_lines.append("".join(current_line_chars)) # 완성된 줄을 리스트에 추가
        except StopIteration: # 필터링된 문자를 모두 사용한 경우
            # 남은 높이만큼 빈 줄('-'으로 채워진 줄)을 추가합니다.
            while len(reconstructed_map_lines) < map_height:
                reconstructed_map_lines.append('-' * map_width)
        # 재구성된 문자열 라인들을 다시 문자 리스트의 리스트로 변환합니다 (ensure_rules_on_map_list 입력 형식).
        final_map_list = [list(line) for line in reconstructed_map_lines]

    # 모든 규칙을 최종 맵 리스트에 적용합니다.
    ensure_rules_on_map_list(final_map_list, map_width, map_height, prompt_str)
            
    # 최종적으로 처리된 맵 리스트를 다시 하나의 문자열로 합쳐 반환합니다.
    return "\n".join("".join(row_list) for row_list in final_map_list)

# 모델을 사용하여 프롬프트를 기반으로 맵 텍스트를 생성하는 함수입니다.
def generate_map(model, tokenizer, prompt, max_length=1024, device="cpu",
                 temperature=0.7, top_k=50, top_p=0.95, num_return_sequences=1,
                 generate_with_prompt=True):
    """
    주어진 모델, 토크나이저, 프롬프트를 사용하여 맵 텍스트를 생성합니다.

    Args:
        model: 학습된 GPT-2 모델.
        tokenizer: GPT-2 토크나이저.
        prompt (str): 맵 생성을 위한 사용자 입력 프롬프트.
        max_length (int, optional): 생성할 텍스트의 최대 길이 (프롬프트 포함). 기본값 1024.
        device (str, optional): 모델 실행 장치 ("cpu" 또는 "cuda"). 기본값 "cpu".
        temperature (float, optional): 샘플링 온도. 높을수록 무작위성이 커짐. 기본값 0.7.
        top_k (int, optional): Top-k 샘플링에서 고려할 상위 k개 토큰. 기본값 50.
        top_p (float, optional): Top-p (Nucleus) 샘플링에서 고려할 누적 확률. 기본값 0.95.
        num_return_sequences (int, optional): 생성할 독립적인 시퀀스(맵)의 수. 기본값 1.
        generate_with_prompt (bool, optional): True면 프롬프트를 포함하여 생성, False면 <MAP> 태그부터 생성. 기본값 True.

    Returns:
        list: 생성된 맵 텍스트(특수 토큰 표현 포함)들의 리스트.
    """
    model.to(device) # 모델을 지정된 장치로 이동합니다.
    model.eval()     # 모델을 평가 모드로 설정합니다 (dropout 등 비활성화).

    # 프롬프트 포함 여부에 따라 입력 텍스트를 구성합니다.
    if generate_with_prompt:
        input_text = f"<PROMPT>{prompt}</PROMPT>\n<MAP>\n"
    else: # 프롬프트 없이 바로 맵 생성 시작 (주로 디버깅이나 특정 조건 테스트용)
        input_text = f"<MAP>\n"
    
    # 입력 텍스트를 토큰 ID로 인코딩하고 PyTorch 텐서로 변환하여 장치로 보냅니다.
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # 맵 생성이 끝나는 지점을 나타내는 토큰 ID를 설정합니다. '</MAP>'을 우선 사용합니다.
    eos_token_for_generation = tokenizer.convert_tokens_to_ids("</MAP>")
    # 만약 '</MAP>'이 어휘에 없다면(UNK 토큰이라면), 토크나이저의 기본 EOS 토큰을 사용합니다.
    if eos_token_for_generation == tokenizer.unk_token_id:
        eos_token_for_generation = tokenizer.eos_token_id
    
    # 모델 생성 시 원치 않는 토큰(예: 기본 EOS 토큰이 '</MAP>'과 다를 경우) 생성을 막기 위한 설정입니다.
    default_eos_token_id = tokenizer.eos_token_id 
    forbidden_tokens_ids_list = []
    if default_eos_token_id is not None and default_eos_token_id != tokenizer.unk_token_id and \
       eos_token_for_generation != default_eos_token_id:
        forbidden_tokens_ids_list.append([default_eos_token_id])

    # 모델의 generate 함수를 호출하여 텍스트를 생성합니다.
    output = model.generate(
        input_ids,
        max_length=max_length,                      # 생성될 텍스트의 최대 길이
        num_return_sequences=num_return_sequences,  # 반환할 시퀀스(맵) 수
        no_repeat_ngram_size=2,                     # 반복되는 n-gram 방지 (2-gram 반복 금지)
        top_k=top_k,                                # Top-k 샘플링
        top_p=top_p,                                # Top-p (Nucleus) 샘플링
        temperature=temperature,                    # 샘플링 온도
        do_sample=True,                             # 샘플링 사용 여부 (False면 그리디 디코딩)
        pad_token_id=tokenizer.pad_token_id,        # 패딩 토큰 ID 설정
        eos_token_id=eos_token_for_generation,      # 시퀀스 종료 토큰 ID 설정 ('</MAP>')
        bad_words_ids=forbidden_tokens_ids_list if forbidden_tokens_ids_list else None # 금지할 토큰 ID 리스트
    )
    
    generated_map_texts_mangled = [] # 생성된 맵 텍스트(특수 토큰 표현 유지)를 저장할 리스트
    # 생성된 각 시퀀스에 대해
    for i in range(num_return_sequences):
        # 토큰 ID 시퀀스를 다시 텍스트로 디코딩합니다. (특수 토큰도 유지)
        generated_text_full = tokenizer.decode(output[i], skip_special_tokens=False)
        map_content_mangled = ""
        # 생성된 전체 텍스트에서 "<MAP>" 태그 이후, "</MAP>" 태그 이전의 내용을 추출합니다.
        if "<MAP>" in generated_text_full:
            parts = generated_text_full.split("<MAP>", 1) # "<MAP>"을 기준으로 한 번만 분리
            if len(parts) > 1: # "<MAP>" 태그가 존재하면
                map_part = parts[1] # "<MAP>" 이후 부분
                if "</MAP>" in map_part: # "</MAP>" 태그가 있으면 그 이전까지 추출
                    map_content_mangled = map_part.split("</MAP>", 1)[0].strip()
                else: # "</MAP>" 태그가 없으면 <MAP> 이후 전부를 사용 (생성 중단된 경우)
                    map_content_mangled = map_part.strip() 
        else: # "<MAP>" 태그 자체가 생성되지 않은 경우 (매우 드문 경우)
            map_content_mangled = generated_text_full 
        generated_map_texts_mangled.append(map_content_mangled)
        
    return generated_map_texts_mangled

# 최종 생성된 맵 텍스트를 콘솔에 보기 좋게 출력하는 함수입니다.
def display_map(map_text_final):
    """
    필터링 및 규칙 적용이 완료된 최종 맵 텍스트를 콘솔에 출력합니다.

    Args:
        map_text_final (str): 최종 맵 텍스트.
    """
    print("\n생성된 맵:")
    print("=" * (MAP_WIDTH if MAP_WIDTH > 0 else 40)) # 맵 너비만큼 구분선 출력
    print(map_text_final)
    print("=" * (MAP_WIDTH if MAP_WIDTH > 0 else 40))

# 최종 생성된 맵 텍스트를 파일로 저장하는 함수입니다.
def save_map_to_file(map_text_final, prompt_info, output_dir="generated_maps"):
    """
    최종 맵 텍스트를 지정된 디렉토리에 파일로 저장합니다.
    파일 이름에는 프롬프트 정보와 타임스탬프가 포함됩니다.

    Args:
        map_text_final (str): 저장할 최종 맵 텍스트.
        prompt_info (str): 파일 이름에 사용될 프롬프트 관련 정보.
        output_dir (str, optional): 맵 파일을 저장할 디렉토리. 기본값 "generated_maps".

    Returns:
        str: 저장된 파일의 전체 경로.
    """
    os.makedirs(output_dir, exist_ok=True) # 저장 디렉토리가 없으면 생성합니다.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # 현재 시간을 문자열로 포맷팅합니다.
    # 프롬프트 정보를 파일 이름에 안전하게 사용할 수 있도록 처리합니다 (특수문자 제거, 길이 제한).
    safe_prompt_filename = "".join(c if c.isalnum() or c in [' ', '_', '-'] else '_' for c in prompt_info).rstrip('_')
    safe_prompt_filename = safe_prompt_filename[:50] # 최대 50자로 제한
    
    filename = f"{output_dir}/gen_map_{safe_prompt_filename}_{timestamp}.txt"
    
    # 파일에 저장될 프롬프트 정보 (예: "_sample1" 같은 접미사 제거)
    base_prompt_for_file = prompt_info.rsplit('_sample', 1)[0] if '_sample' in prompt_info else prompt_info
    
    # 파일에 프롬프트와 맵 데이터를 지정된 형식으로 저장합니다.
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"<PROMPT>{base_prompt_for_file}</PROMPT>\n<MAP>\n{map_text_final}\n</MAP>")
    print(f"생성된 맵이 {filename}에 저장되었습니다.")
    return filename

# 이 스크립트가 직접 실행될 때 호출되는 메인 부분입니다.
if __name__ == "__main__":
    # 명령줄 인자를 파싱하기 위한 ArgumentParser 객체를 생성합니다.
    parser = argparse.ArgumentParser(description="학습된 GPT-2 모델을 사용하여 게임 맵 생성하기")
    parser.add_argument("--prompt", type=str, default="medium difficulty map, some traps", help="맵 생성을 위한 기본 프롬프트")
    parser.add_argument("--model_dir", type=str, default="trained_model", help="학습된 모델이 저장된 디렉토리")
    parser.add_argument("--output_dir", type=str, default="generated_maps", help="생성된 맵 파일을 저장할 디렉토리")
    parser.add_argument("--max_length", type=int, default=1024, help="모델이 생성할 텍스트의 최대 길이 (프롬프트 포함). 실제 맵 크기보다 충분히 커야 함.")
    parser.add_argument("--temperature", type=float, default=0.8, help="생성 다양성을 위한 온도 (0.7 ~ 0.9 권장)")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k 샘플링 파라미터")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p (Nucleus) 샘플링 파라미터")
    parser.add_argument("--count", type=int, default=3, help="생성할 맵의 개수")
    parser.add_argument("--no_prompt", action="store_true", help="프롬프트 없이 '<MAP>' 태그부터 바로 생성 시작 (테스트용)")
    args = parser.parse_args() # 명령줄 인자를 파싱합니다.

    # GPU 사용 가능 여부를 확인하고, 가능하면 GPU(cuda)를, 아니면 CPU(cpu)를 사용합니다.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    # 학습된 모델과 토크나이저를 로드합니다.
    model, tokenizer = load_trained_model(args.model_dir)
    if model is None or tokenizer is None: exit(1) # 로드 실패 시 종료합니다.

    should_generate_with_prompt = not args.no_prompt # 프롬프트 사용 여부 결정
    if should_generate_with_prompt:
        print(f"프롬프트: '{args.prompt}'로 {args.count}개의 맵 생성 중 (max_length={args.max_length}, temp={args.temperature})...")
    else:
        print(f"프롬프트 없이 {args.count}개의 맵 생성 중 (max_length={args.max_length}, temp={args.temperature})...")
    
    # 설정된 파라미터로 맵 생성을 요청합니다.
    generated_maps_mangled = generate_map(
        model, tokenizer, args.prompt, 
        max_length=args.max_length, device=device,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        num_return_sequences=args.count, generate_with_prompt=should_generate_with_prompt
    )
    
    # 생성된 각 맵에 대해 후처리 및 저장 과정을 수행합니다.
    for i, mangled_map_content in enumerate(generated_maps_mangled):
        # 1. 특수 토큰 표현을 원래 문자로 되돌립니다.
        reverted_map_content = revert_map_representation(mangled_map_content)
        
        # 규칙 적용 시 사용할 프롬프트를 설정합니다 (T 규칙 등).
        current_prompt_for_rules = args.prompt if should_generate_with_prompt else ""
        
        # 2. 맵 필터링, 재구성 및 모든 게임 규칙을 적용합니다.
        final_structured_map = filter_and_reconstruct_map(
            reverted_map_content, MAP_WIDTH, MAP_HEIGHT, current_prompt_for_rules
        )
        
        # 3. 최종 맵을 콘솔에 출력합니다.
        display_map(final_structured_map)
        
        # 4. 최종 맵을 파일로 저장합니다. 파일 이름용 프롬프트 정보를 만듭니다.
        save_prompt_info = f"{args.prompt}_sample{i+1}" if should_generate_with_prompt else f"no_prompt_sample{i+1}"
        save_map_to_file(final_structured_map, save_prompt_info, args.output_dir)
            
    print(f"\n{len(generated_maps_mangled)}개의 맵 생성이 완료되었습니다. '{args.output_dir}' 폴더를 확인하세요.")