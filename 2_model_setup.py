import torch # PyTorch 라이브러리, 텐서 연산 및 신경망 모델 구축에 사용됩니다.
from torch.utils.data import Dataset # PyTorch에서 사용자 정의 데이터셋을 만들기 위한 기본 클래스입니다.
from transformers import GPT2LMHeadModel, GPT2Tokenizer # Hugging Face Transformers 라이브러리에서 GPT-2 언어 모델과 토크나이저를 가져옵니다. distilgpt2도 GPT2Tokenizer를 사용합니다.
import json # JSON 형식의 데이터를 읽고 쓰기 위한 모듈입니다.
import os # 운영체제와 상호작용하기 위한 모듈로, 파일 경로 생성 등에 사용됩니다.

# PyTorch의 Dataset 클래스를 상속받아 맵 데이터셋을 정의합니다.
class MapDataset(Dataset):
    """
    학습 및 평가에 사용될 맵 데이터를 처리하는 사용자 정의 데이터셋 클래스입니다.
    프롬프트와 맵 데이터를 입력으로 받아 모델이 학습할 수 있는 형태로 변환합니다.
    """
    # 데이터셋 객체 초기화 시 호출됩니다.
    def __init__(self, maps, tokenizer, max_length=1024):
        """
        MapDataset 객체를 초기화합니다.

        Args:
            maps (list): 프롬프트와 맵 텍스트를 포함하는 딕셔너리들의 리스트입니다.
            tokenizer: 입력 텍스트를 토큰 ID로 변환하는 데 사용될 토크나이저 객체입니다.
            max_length (int, optional): 토큰화된 입력의 최대 길이입니다. 이 길이를 초과하는 시퀀스는 잘리고, 부족하면 패딩됩니다. 기본값은 1024입니다.
        """
        self.maps = maps # 맵 데이터 리스트를 저장합니다.
        self.tokenizer = tokenizer # 토크나이저를 저장합니다.
        self.max_length = max_length # 최대 길이를 저장합니다.

    # 데이터셋의 총 샘플 수를 반환합니다.
    def __len__(self):
        """
        데이터셋에 포함된 총 맵 아이템의 수를 반환합니다.
        """
        return len(self.maps)

    # 주어진 인덱스(idx)에 해당하는 데이터 샘플을 반환합니다.
    def __getitem__(self, idx):
        """
        지정된 인덱스에 해당하는 맵 아이템을 가져와 모델 입력 형식으로 변환합니다.

        Args:
            idx (int): 가져올 맵 아이템의 인덱스입니다.

        Returns:
            dict: 'input_ids', 'attention_mask', 'labels'를 키로 하는 딕셔너리입니다.
                  각 값은 PyTorch 텐서 형태입니다.
        """
        map_item = self.maps[idx] # 해당 인덱스의 맵 아이템(딕셔너리)을 가져옵니다.
        prompt = map_item["prompt"] # 프롬프트 부분을 추출합니다.
        map_text = map_item["map"]  # 이미 '[P]', '[W]' 등으로 변환된 맵 텍스트를 추출합니다.

        # 프롬프트 유무에 따라 모델에 입력될 전체 텍스트를 구성합니다.
        # <PROMPT>, </PROMPT>, <MAP>, </MAP> 태그를 사용하여 구조화합니다.
        if prompt:
            text = f"<PROMPT>{prompt}</PROMPT>\n<MAP>\n{map_text}\n</MAP>"
        else:
            text = f"<MAP>\n{map_text}\n</MAP>"

        # 토크나이저를 사용하여 텍스트를 토큰 ID 시퀀스로 변환합니다.
        # truncation=True: max_length를 초과하면 자릅니다.
        # padding="max_length": max_length까지 패딩합니다.
        # return_tensors="pt": PyTorch 텐서로 반환합니다.
        encodings = self.tokenizer(text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")

        # 토크나이저 출력에서 input_ids와 attention_mask를 추출하고, 불필요한 차원을 제거(squeeze)합니다.
        input_ids = encodings.input_ids.squeeze()
        attention_mask = encodings.attention_mask.squeeze()
        # 언어 모델 학습 시에는 보통 input_ids를 복제하여 labels로 사용합니다. (다음 토큰 예측)
        labels = input_ids.clone()

        # 모델 학습에 필요한 형태로 딕셔너리에 담아 반환합니다.
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# 사전 학습된 distilgpt2 모델과 토크나이저를 설정하고, 사용자 정의 토큰을 추가하는 함수입니다.
def setup_model_and_tokenizer():
    """
    Hugging Face Transformers 라이브러리를 사용하여 'distilgpt2' 모델과
    관련 토크나이저를 로드하고, 맵 생성 작업에 필요한 사용자 정의 토큰들을
    토크나이저 어휘에 추가한 후 모델의 임베딩 레이어를 조정합니다.

    Returns:
        tuple: (model, tokenizer) 튜플입니다.
               모델 또는 토크나이저 로드에 실패하면 (None, None)을 반환합니다.
    """
    
    model_name = "distilgpt2" # 사용할 사전 학습 모델의 이름입니다.
    try:
        # 지정된 모델 이름으로 토크나이저와 언어 모델(LMHeadModel)을 로드합니다.
        # distilgpt2는 GPT2Tokenizer와 호환됩니다.
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        print(f"성공적으로 '{model_name}' 모델과 토크나이저를 로드했습니다.")
    except Exception as e:
        # 모델/토크나이저 로드 중 오류 발생 시 메시지를 출력하고 None을 반환합니다.
        print(f"'{model_name}' 로드 중 오류 발생: {e}")
        print("인터넷 연결을 확인하거나, 모델 이름이 정확한지 확인해주세요.")
        print("스크립트를 중단합니다.")
        return None, None

    # 1_data_preparation.py에서 정의된 맵 타일 표현과 줄바꿈 문자를 정의합니다.
    # 이 문자들은 토크나이저에 특별 토큰으로 추가될 것입니다.
    map_tile_chars = ['[P]', '[W]', '[-]', '[G]', '[O]', '[T]', '\n'] 
    
    # 데이터에서 사용될 특별 태그들을 정의합니다.
    special_tags = ["<PROMPT>", "</PROMPT>", "<MAP>", "</MAP>"]
    
    # 1. 패딩 토큰(<PAD>) 추가:
    # DistilGPT2/GPT2 모델은 기본적으로 pad_token이 설정되어 있지 않을 수 있습니다.
    # 패딩은 배치 처리 시 시퀀스 길이를 통일하기 위해 필요합니다.
    if tokenizer.pad_token is None:
        # '[PAD]'라는 문자열로 새로운 패딩 토큰을 추가합니다.
        # 기존 어휘와의 충돌을 피하기 위해 대괄호로 감싼 형태를 사용합니다.
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        print(f"Added [PAD] token. New vocab size for {model_name}: {len(tokenizer)}")
    
    # 2. 사용자 정의 토큰들(맵 타일 표현 + 태그)을 특별 토큰으로 추가:
    # 이 토큰들이 하나의 단위로 인식되도록 `additional_special_tokens`로 추가합니다.
    # 중복을 제거하기 위해 set으로 변환 후 다시 list로 만듭니다.
    all_our_special_tokens = list(set(special_tags + map_tile_chars))
    
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": all_our_special_tokens})
    print(f"additional_special_tokens로 추가 시도한 토큰 목록 ({len(all_our_special_tokens)}개): {sorted(all_our_special_tokens)}")
    print(f"add_special_tokens 후 실제로 어휘에 더해진 토큰 수: {num_added}")

    # 3. 어휘 크기 변경에 맞춰 모델의 토큰 임베딩 레이어 크기 조정:
    # 새로운 토큰이 추가되었으므로, 모델이 이 토큰들을 이해할 수 있도록 임베딩 레이어의 크기를 늘립니다.
    model.resize_token_embeddings(len(tokenizer))
    
    # 4. 모델 설정(config)에 패딩 토큰 ID 설정:
    # 모델의 `generate` 함수 등에서 패딩 토큰 ID를 참조하므로, 이를 명시적으로 설정해줍니다.
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    else:
        # 만약 `tokenizer.pad_token_id`가 자동으로 설정되지 않았다면,
        # 직접 '[PAD]' 또는 '<PAD>' 문자열을 ID로 변환하여 설정합니다.
        pad_token_str = "[PAD]" if "[PAD]" in tokenizer.get_vocab() else "<PAD>"
        model.config.pad_token_id = tokenizer.convert_tokens_to_ids(pad_token_str)
        # 만약 변환된 ID가 UNK(unknown) 토큰 ID와 같다면, 패딩 토큰이 제대로 추가되지 않은 것입니다.
        if model.config.pad_token_id == tokenizer.unk_token_id:
            print(f"경고: PAD 토큰 ID를 찾을 수 없습니다 ({pad_token_str}). 패딩 관련 문제가 발생할 수 있습니다.")

    # 토크나이저 설정 후 상세 디버깅 정보를 출력합니다.
    print("\n--- Tokenizer 설정 후 상세 디버깅 ---")
    print(f"사용된 모델: {model_name}")
    print(f"사용된 map_tile_chars (새로운 표현): {map_tile_chars}")
    print(f"사용된 special_tags: {special_tags}")
    print(f"최종 Tokenizer 어휘 크기: {len(tokenizer)}")
    print(f"Pad token string: '{tokenizer.pad_token}', Pad token ID from model config: {model.config.pad_token_id}")

    # 주요 의도된 토큰들이 올바르게 인식되고, 새로운 ID를 할당받았는지 확인합니다.
    print("\n주요 의도 토큰 ID 확인 (모두 새로운 special token이어야 함):")
    # 확인할 토큰 목록은 우리가 추가한 모든 특별 토큰과 패딩 토큰입니다.
    tokens_to_verify = list(set(all_our_special_tokens + [tokenizer.pad_token]))

    # GPT-2/DistilGPT2의 기본 어휘 크기는 대략 50257입니다.
    # 새로 추가된 토큰들은 이보다 큰 ID를 가져야 합니다.
    base_vocab_size_approx = 50257 

    for token_str in sorted(tokens_to_verify):
        if token_str is None : continue # pad_token이 None일 경우 건너뜁니다.
        try:
            token_id = tokenizer.convert_tokens_to_ids(token_str) # 토큰 문자열을 ID로 변환합니다.
            decoded_from_id = tokenizer.decode([token_id]) # ID를 다시 문자열로 디코딩하여 확인합니다.
            note = ""
            # ID가 기본 어휘 크기보다 크고 UNK 토큰이 아니면, 새로 추가된 특별 토큰으로 간주합니다.
            if token_id >= base_vocab_size_approx and token_id != tokenizer.unk_token_id:
                note = "(새로 추가된 special token 영역으로 추정)"
            # UNK 토큰이 아니지만 ID가 낮은 경우, 기존 어휘와 겹치거나 의도와 다를 수 있습니다.
            elif token_id != tokenizer.unk_token_id:
                note = "(기존 어휘 또는 낮은 ID의 special token - 의도와 다를 수 있음!)"
            else: # UNK 토큰인 경우, 어휘에 제대로 추가되지 않은 것입니다.
                note = "(UNK 토큰 - 어휘에 없음!)"
            
            print(f"  '{token_str}' -> ID: {token_id}, Decoded: '{decoded_from_id}' {note}")
        except Exception as e:
            print(f"  '{token_str}' -> ID 확인 중 오류: {e}")
    
    # 이전 단일 문자 타일들이 이제는 UNK (Unknown) 토큰으로 처리되는지 확인합니다.
    # (예: 'P'가 아니라 '[P]'가 하나의 토큰으로 인식되어야 함)
    print("\n이전 단일 문자 타일들의 현재 상태 확인 (UNK여야 함):")
    old_map_tiles_single_char = ['P', 'W', '-', 'T', 'G', 'O']
    for old_tile in old_map_tiles_single_char:
        token_id = tokenizer.convert_tokens_to_ids(old_tile)
        decoded_from_id = tokenizer.decode([token_id])
        status = "UNK" if token_id == tokenizer.unk_token_id else "기존 어휘에 존재함 (의도와 다름!)"
        print(f"  이전 타일 '{old_tile}' -> ID: {token_id}, Decoded: '{decoded_from_id}', 상태: {status}")

    return model, tokenizer # 설정된 모델과 토크나이저를 반환합니다.

# 설정된 모델과 토크나이저를 지정된 디렉토리에 저장하는 함수입니다.
def save_model_and_tokenizer(model, tokenizer, save_dir="model_setup"):
    """
    학습된 모델과 토크나이저를 지정된 디렉토리에 저장합니다.

    Args:
        model: 저장할 Hugging Face 모델 객체입니다.
        tokenizer: 저장할 Hugging Face 토크나이저 객체입니다.
        save_dir (str, optional): 모델과 토크나이저를 저장할 디렉토리 이름입니다. 기본값은 "model_setup"입니다.
    """
    # 모델 또는 토크나이저가 None이면 저장하지 않고 메시지를 출력합니다.
    if model is None or tokenizer is None:
        print(f"{save_dir}에 저장할 모델 또는 토크나이저가 없습니다 (None).")
        return
    # 저장할 디렉토리가 없으면 생성합니다.
    os.makedirs(save_dir, exist_ok=True)
    # 모델과 토크나이저를 지정된 디렉토리에 저장합니다.
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"모델과 토크나이저가 {save_dir}에 저장되었습니다.")

# 1_data_preparation.py에서 준비된 학습 및 검증 데이터셋을 로드하는 함수입니다.
def load_datasets():
    """
    'data' 폴더에 저장된 'train_maps.json'과 'val_maps.json' 파일을 로드합니다.
    이 파일들은 1_data_preparation.py 스크립트를 통해 생성됩니다.

    Returns:
        tuple: (train_maps, val_maps) 튜플입니다.
               파일 로드에 실패하면 (None, None)을 반환합니다.
    """
    try:
        # 학습 데이터를 JSON 파일에서 로드합니다.
        with open("data/train_maps.json", "r", encoding="utf-8") as f:
            train_maps = json.load(f)
        # 검증 데이터를 JSON 파일에서 로드합니다.
        with open("data/val_maps.json", "r", encoding="utf-8") as f:
            val_maps = json.load(f)
        return train_maps, val_maps
    except FileNotFoundError:
        # 파일이 존재하지 않을 경우 오류 메시지를 출력하고 None을 반환합니다.
        print("데이터셋 파일(train_maps.json 또는 val_maps.json)이 'data' 폴더에 없습니다.")
        print("먼저 1_data_preparation.py를 성공적으로 실행해주세요.")
        return None, None

# 이 스크립트가 직접 실행될 때 수행되는 메인 로직입니다.
if __name__ == "__main__":
    # 준비된 학습 및 검증 데이터셋을 로드합니다.
    train_maps, val_maps = load_datasets()
    # 데이터셋 로드에 실패하면 스크립트를 중단합니다.
    if train_maps is None or val_maps is None:
        print("데이터셋 로드 실패. 스크립트를 중단합니다.")
        exit(1) 

    print("모델 및 토크나이저 설정 시작...")
    # 모델과 토크나이저를 설정하고 사용자 정의 토큰을 추가합니다. (디버깅 출력 포함)
    model, tokenizer = setup_model_and_tokenizer() 
    
    # 모델 또는 토크나이저 설정에 실패하면 스크립트를 중단합니다.
    if model is None or tokenizer is None:
        print("모델 또는 토크나이저 설정 실패. 스크립트를 중단합니다.")
        exit(1)

    # 설정된 모델과 토크나이저를 파일로 저장합니다.
    save_model_and_tokenizer(model, tokenizer)

    # 데이터셋 샘플을 생성하고 토크나이저로 디코딩하여 확인하는 테스트 코드입니다. (선택 사항)
    # 실제 학습 전에 입력 데이터가 올바르게 처리되는지 확인하기 유용합니다.
    if train_maps:
        # MapDataset에 전달할 max_length는 실제 데이터의 평균/최대 토큰 길이를 고려하여 설정합니다.
        # distilgpt2의 최대 컨텍스트 길이는 1024입니다.
        sample_map_for_dataset_test = [train_maps[0]] if train_maps else [] # 첫 번째 학습 데이터를 샘플로 사용합니다.
        if sample_map_for_dataset_test:
            dataset_max_length = 1024 # 테스트용 MapDataset의 max_length를 설정합니다.
            test_dataset = MapDataset(sample_map_for_dataset_test, tokenizer, max_length=dataset_max_length) 
            if len(test_dataset) > 0:
                sample = test_dataset[0] # 첫 번째 샘플을 가져옵니다.
                print("\n--- 데이터셋 입력 ID 디버깅 (첫 번째 학습 샘플) ---")
                
                # 입력 ID의 일부(처음 100개)를 출력합니다.
                input_ids_subset = sample['input_ids'][:100] 
                print(f"입력 ID (처음 {len(input_ids_subset)}개, max_length={dataset_max_length}): {input_ids_subset.tolist()}")
                
                # 토큰 ID 시퀀스를 다시 텍스트로 디코딩하여 출력합니다. (너무 길면 일부만)
                print("\nTokenizer로 디코딩된 전체 입력 텍스트 (일부만 표시될 수 있음):")
                decoded_text = tokenizer.decode(sample['input_ids'])
                print(decoded_text[:1000] + "..." if len(decoded_text) > 1000 else decoded_text)

    print("\n2단계 완료: 모델과 토크나이저 설정 및 디버깅 정보 출력이 끝났습니다.")