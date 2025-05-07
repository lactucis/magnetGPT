import os # 운영 체제와 상호 작용하기 위한 모듈입니다. 파일 경로 생성, 디렉토리 확인 등에 사용됩니다.
import json # JSON 데이터를 다루기 위한 모듈입니다. 학습 통계 등을 저장하고 불러올 때 사용됩니다.
import torch # PyTorch 라이브러리입니다. 텐서 연산, 신경망 모델, GPU 가속 등을 지원합니다.
from torch.utils.data import DataLoader # PyTorch에서 데이터를 배치 단위로 효율적으로 로드하기 위한 클래스입니다.
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup # Hugging Face Transformers 라이브러리의 주요 구성 요소들입니다.
# GPT2LMHeadModel: GPT-2 언어 모델 (다음 토큰 예측 기능 포함)
# GPT2Tokenizer: GPT-2 모델용 토크나이저
# AdamW: 가중치 감쇠(weight decay)가 수정된 Adam 옵티마이저
# get_linear_schedule_with_warmup: 학습률 스케줄러 (초기에는 학습률을 점진적으로 증가시키고 이후 선형적으로 감소)
from tqdm import tqdm # 반복문 진행 상황을 시각적으로 보여주는 라이브러리입니다.
import numpy as np # 수치 연산을 위한 라이브러리입니다. 평균 계산 등에 사용됩니다.
from pathlib import Path # 파일 및 디렉토리 경로를 객체 지향적으로 다루기 위한 모듈입니다.

# 2_model_setup.py 파일에서 정의한 MapDataset 클래스를 가져옵니다.
# 이 클래스는 맵 데이터를 모델 학습에 적합한 형태로 변환하는 역할을 합니다.
from model_setup import MapDataset

# 저장된 모델과 토크나이저를 로드하는 함수입니다.
def load_model_and_tokenizer(model_dir="model_setup"):
    """
    지정된 디렉토리에서 사전 설정된 모델과 토크나이저를 로드합니다.
    이 함수는 2_model_setup.py에서 모델과 토크나이저를 저장한 후 사용됩니다.

    Args:
        model_dir (str, optional): 모델과 토크나이저가 저장된 디렉토리 경로. 기본값은 "model_setup"입니다.

    Returns:
        tuple: (model, tokenizer) 튜플. 로드 실패 시 (None, None)을 반환합니다.
    """
    try:
        # 지정된 디렉토리에서 토크나이저를 로드합니다.
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        # 지정된 디렉토리에서 모델을 로드합니다.
        model = GPT2LMHeadModel.from_pretrained(model_dir)
        return model, tokenizer
    except Exception as e:
        # 로드 중 오류가 발생하면 오류 메시지를 출력하고 None을 반환합니다.
        print(f"모델 로드 중 오류 발생: {e}")
        print("먼저 2_model_setup.py를 실행하여 model_setup 디렉토리에 모델과 토크나이저를 저장해야 합니다.")
        return None, None

# 1_data_preparation.py에서 준비된 학습 및 검증 데이터셋을 JSON 파일에서 로드하는 함수입니다.
def load_datasets():
    """
    'data' 폴더에 저장된 'train_maps.json'과 'val_maps.json' 파일을 로드합니다.
    이 파일들은 1_data_preparation.py 스크립트를 통해 생성된 전처리된 맵 데이터입니다.

    Returns:
        tuple: (train_maps, val_maps) 튜플. 파일 로드 실패 시 (None, None)을 반환합니다.
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
        print("먼저 1_data_preparation.py를 성공적으로 실행하여 데이터셋을 준비해주세요.")
        return None, None

# 모델을 학습시키는 메인 함수입니다.
def train_model(model, train_dataset, val_dataset, device, output_dir="trained_model", 
                num_epochs=10, batch_size=4, learning_rate=5e-5):
    """
    주어진 모델과 데이터셋을 사용하여 모델을 학습시킵니다.

    Args:
        model: 학습할 PyTorch 모델 객체입니다.
        train_dataset: 학습용 MapDataset 객체입니다.
        val_dataset: 검증용 MapDataset 객체입니다.
        device: 학습에 사용할 장치 ('cuda' 또는 'cpu')입니다.
        output_dir (str, optional): 학습된 모델과 통계 정보를 저장할 디렉토리. 기본값은 "trained_model"입니다.
        num_epochs (int, optional): 총 학습 에폭 수. 기본값은 10입니다.
        batch_size (int, optional): 학습 및 검증 시 사용할 배치 크기. 기본값은 4입니다.
        learning_rate (float, optional): 옵티마이저의 학습률. 기본값은 5e-5입니다.

    Returns:
        tuple: (trained_model, training_stats) 튜플.
               trained_model은 학습 완료된 모델, training_stats는 에폭별 학습/검증 손실을 담은 리스트입니다.
    """
    # 학습된 모델을 저장할 디렉토리를 생성합니다. 이미 존재하면 오류를 발생시키지 않습니다.
    os.makedirs(output_dir, exist_ok=True)
    
    # 학습 데이터 로더를 생성합니다. 데이터를 섞고(shuffle=True) 배치 크기만큼 가져옵니다.
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # 검증 데이터 로더를 생성합니다. 검증 시에는 데이터를 섞지 않습니다.
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    # AdamW 옵티마이저를 설정합니다. 모델의 파라미터와 학습률을 전달합니다.
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # 학습률 스케줄러를 설정합니다.
    # 총 학습 스텝 수를 계산합니다 (학습 데이터 로더의 길이 * 총 에폭 수).
    total_steps = len(train_dataloader) * num_epochs
    # Warmup 스텝 없이, 전체 학습 스텝 동안 학습률을 선형적으로 감소시키는 스케줄러를 사용합니다.
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, # 초기 학습률 증가 단계 없음
        num_training_steps=total_steps
    )
    
    # 모델을 지정된 장치(GPU 또는 CPU)로 이동시킵니다.
    model.to(device)
    
    # 에폭별 학습 및 검증 손실을 기록하기 위한 리스트입니다.
    training_stats = []
    
    # 지정된 에폭 수만큼 학습 루프를 실행합니다.
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # 모델을 학습 모드로 설정합니다. (Dropout 등 활성화)
        model.train()
        train_losses = [] # 현재 에폭의 학습 손실을 저장할 리스트입니다.
        
        # tqdm을 사용하여 학습 진행 상황을 표시합니다.
        progress_bar = tqdm(train_dataloader, desc=f"학습 중 (Epoch {epoch+1})")
        
        # 학습 데이터 로더에서 배치 단위로 데이터를 가져와 학습을 진행합니다.
        for batch in progress_bar:
            # 배치에서 input_ids, attention_mask, labels를 가져와 지정된 장치로 이동시킵니다.
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # 모델에 입력을 전달하여 출력을 계산합니다. labels를 함께 전달하면 손실(loss)이 자동으로 계산됩니다.
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss # 계산된 손실 값을 가져옵니다.
            train_losses.append(loss.item()) # 현재 배치의 손실을 리스트에 추가합니다. .item()은 텐서에서 숫자 값을 추출합니다.
            
            # 역전파: 손실에 대한 그래디언트를 계산합니다.
            loss.backward()
            # 옵티마이저 스텝: 계산된 그래디언트를 사용하여 모델 파라미터를 업데이트합니다.
            optimizer.step()
            # 스케줄러 스텝: 학습률을 조정합니다.
            scheduler.step()
            # 그래디언트 초기화: 다음 배치를 위해 이전 그래디언트를 지웁니다.
            optimizer.zero_grad()
            
            # tqdm 진행 바에 현재 배치의 손실 값을 표시합니다.
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # 현재 에폭의 평균 학습 손실을 계산하여 출력합니다.
        avg_train_loss = np.mean(train_losses)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # 모델을 평가 모드로 설정합니다. (Dropout 등 비활성화)
        model.eval()
        val_losses = [] # 현재 에폭의 검증 손실을 저장할 리스트입니다.
        
        # 그래디언트 계산을 비활성화합니다. (검증 시에는 파라미터 업데이트가 필요 없음)
        with torch.no_grad():
            # 검증 데이터 로더에서 배치 단위로 데이터를 가져와 검증을 진행합니다.
            for batch in tqdm(val_dataloader, desc="검증 중"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                val_losses.append(loss.item())
        
        # 현재 에폭의 평균 검증 손실을 계산하여 출력합니다.
        avg_val_loss = np.mean(val_losses)
        print(f"Average validation loss: {avg_val_loss:.4f}")
        
        # 현재 에폭의 학습 통계를 기록합니다.
        training_stats.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        })
        
        # 매 에폭마다 모델을 중간 저장합니다.
        # 에폭별 디렉토리를 생성합니다 (예: trained_model/epoch_1).
        epoch_dir = os.path.join(output_dir, f"epoch_{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)
        model.save_pretrained(epoch_dir) # 현재 에폭의 모델 상태를 저장합니다.
    
    # 모든 에폭 학습 완료 후 최종 모델을 저장합니다.
    model.save_pretrained(output_dir)
    
    # 전체 학습 통계를 JSON 파일로 저장합니다.
    with open(os.path.join(output_dir, "training_stats.json"), "w") as f:
        json.dump(training_stats, f, indent=2) # indent=2로 가독성 좋게 저장합니다.
    
    return model, training_stats # 학습된 모델과 학습 통계를 반환합니다.

# 이 스크립트가 직접 실행될 때 수행되는 메인 로직입니다.
if __name__ == "__main__":
    # GPU 사용 가능 여부를 확인하고, 가능하면 GPU(cuda)를, 아니면 CPU(cpu)를 사용합니다.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")
    
    # 2_model_setup.py에서 저장한 모델과 토크나이저를 로드합니다.
    model, tokenizer = load_model_and_tokenizer()
    # 로드에 실패하면 스크립트를 종료합니다.
    if model is None or tokenizer is None:
        exit(1)
    
    # 1_data_preparation.py에서 준비한 학습 및 검증 데이터(맵 정보)를 로드합니다.
    train_maps, val_maps = load_datasets()
    # 데이터 로드에 실패하면 스크립트를 종료합니다.
    if train_maps is None or val_maps is None:
        exit(1)
    
    # 로드한 맵 정보와 토크나이저를 사용하여 MapDataset 객체를 생성합니다.
    # max_length는 MapDataset의 기본값을 사용합니다. 필요시 tokenizer.model_max_length 등을 참고하여 설정할 수 있습니다.
    train_dataset = MapDataset(train_maps, tokenizer)
    val_dataset = MapDataset(val_maps, tokenizer)
    
    # 학습 파라미터를 설정합니다.
    num_epochs = 10    # 총 학습 에폭 수
    batch_size = 4     # 배치 크기 (메모리 상황에 따라 조절)
    learning_rate = 5e-5 # 학습률

    # 모델 학습을 시작합니다.
    print(f"학습 시작: {num_epochs}개의 에폭, 배치 크기 {batch_size}, 학습률 {learning_rate}")
    trained_model, training_stats = train_model(
        model, 
        train_dataset, 
        val_dataset, 
        device, 
        output_dir="trained_model", # 학습된 모델이 저장될 디렉토리
        num_epochs=num_epochs, 
        batch_size=batch_size, 
        learning_rate=learning_rate
    )
    
    # 학습 과정에서 모델은 output_dir (기본값: "trained_model")에 이미 저장되었습니다.
    # 토크나이저는 모델과 함께 변경되지 않았으므로, 초기 로드된 토크나이저를
    # 최종 학습된 모델 디렉토리에 함께 저장하여 일관성을 유지합니다.
    # (만약 토크나이저가 학습 중에 변경되는 로직이 있다면, 변경된 토크나이저를 저장해야 합니다.)
    tokenizer.save_pretrained("trained_model")
    
    print("3단계 완료: 모델 학습이 끝났습니다.")