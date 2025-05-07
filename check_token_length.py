from transformers import GPT2Tokenizer
import json

def get_token_length(text, tokenizer_path="model_setup"): # 저장된 토크나이저 사용
    """주어진 텍스트의 토큰 길이를 계산합니다."""
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        # 토크나이저는 이미 [P], [W] 등을 알고 있어야 합니다.
        token_ids = tokenizer.encode(text)
        return len(token_ids)
    except Exception as e:
        print(f"토크나이저 로드 또는 인코딩 중 오류: {e}")
        return -1

if __name__ == "__main__":
    # data/train_maps.json 또는 data/val_maps.json 에서 맵 데이터 로드
    try:
        with open("data/train_maps.json", "r", encoding="utf-8") as f:
            maps_data = json.load(f)
    except FileNotFoundError:
        print("'data/train_maps.json' 파일을 찾을 수 없습니다. 1_data_preparation.py를 먼저 실행하세요.")
        exit()

    if not maps_data:
        print("맵 데이터가 비어있습니다.")
        exit()

    # 몇 개의 샘플 맵에 대해 토큰 길이 확인
    num_samples_to_check = min(5, len(maps_data))
    max_observed_token_length = 0

    print("맵 샘플들의 예상 토큰 길이 (프롬프트 + 태그 + 맵 내용 포함):")
    for i in range(num_samples_to_check):
        map_item = maps_data[i]
        prompt = map_item["prompt"]
        map_text_mangled = map_item["map"] # 이미 '[P]', '[W]' 등으로 변환된 맵

        # 학습 시 모델에 입력되는 실제 텍스트 구성
        full_text_representation = f"<PROMPT>{prompt}</PROMPT>\n<MAP>\n{map_text_mangled}\n</MAP>"

        token_length = get_token_length(full_text_representation)

        if token_length != -1:
            print(f"  맵 샘플 {i+1} (프롬프트: '{prompt[:20]}...'): {token_length} 토큰")
            if token_length > max_observed_token_length:
                max_observed_token_length = token_length
        else:
            print(f"  맵 샘플 {i+1}의 토큰 길이를 계산할 수 없습니다.")

    print(f"\n확인된 샘플 중 최대 토큰 길이: {max_observed_token_length}")

    # distilgpt2의 최대 컨텍스트 길이
    model_max_context_length = 1024 # distilgpt2
    print(f"DistilGPT2의 일반적인 최대 컨텍스트 길이: {model_max_context_length} 토큰")

    if max_observed_token_length > model_max_context_length:
        print(f"경고: 최대 관찰된 토큰 길이({max_observed_token_length})가 모델의 최대 컨텍스트 길이({model_max_context_length})를 초과할 수 있습니다.")
        print("      이 경우 맵을 더 작은 청크로 나누거나, max_length를 모델 한계 내로 설정해야 합니다.")
    elif max_observed_token_length == 0:
        print("경고: 맵 데이터의 토큰 길이를 제대로 측정하지 못했습니다. 맵 파일과 1_data_preparation.py 출력을 확인해주세요.")
    else:
        suggested_max_length_generate = max_observed_token_length + 20 # 약간의 여유분
        print(f"생성 시 권장되는 'max_length' (샘플 기반): 약 {suggested_max_length_generate} (모델 한계 내에서 조절)")
        print(f"  (MapDataset의 max_length도 이와 유사하게 설정되었는지 확인 필요)")