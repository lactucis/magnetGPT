# MagnetGPT: 텍스트 프롬프트 기반 2D 퍼즐 게임 맵 생성기 (LLM 파인튜닝)

본 프로젝트는 DistilGPT2 모델을 파인튜닝하여, 사용자가 입력한 텍스트 프롬프트에 맞는 2D 퍼즐 게임 맵을 자동으로 생성하는 연구의 구현체입니다. 기존에 개발된 특정 2D 퍼즐 게임의 맵 데이터를 텍스트 시퀀스로 변환하고, 이를 기반으로 새로운 맵을 생성하는 가능성을 탐구합니다.

## 주요 목표

* **프롬프트 기반 맵 생성**: 텍스트 프롬프트(예: "medium map, no traps, platform layout")를 입력으로 받아, 이에 맞는 맵을 생성합니다.
* **LLM 파인튜닝 활용**: DistilGPT2 모델을 특정 게임 맵 데이터에 맞게 파인튜닝하여 도메인 특화된 생성 능력을 학습합니다.
* **텍스트 시퀀스로서의 맵 표현**: 2D 맵의 각 타일을 고유한 문자 토큰으로 표현하고, 이를 긴 텍스트 시퀀스로 변환하여 모델 학습에 사용합니다.
* **플레이 가능성 확보를 위한 후처리**: 생성된 맵에 최소 배치 규칙을 적용하여 플레이 가능한 맵을 생성합니다.
* **데이터 증강 및 토큰화 전략**: 데이터 증강 및 커스텀 토큰 표현을 적용하여 모델 학습의 효율성을 개선합니다.


## 실행 방법 (How to Run)

### 1. 사전 준비 (Prerequisites)

* Python 3.x 환경을 준비합니다.
* 필요한 라이브러리를 설치합니다:
    ```bash
    pip install torch transformers
    ```

### 2. 데이터 준비 (Data Preparation)

* `maps/` 폴더에 원본 맵 데이터 파일을 준비합니다. 각 파일은 다음 형식이어야 합니다:

    ```xml
    <prompt>medium map, no traps</prompt>
    <map>
    WWWWWW
    W--P-W
    W-G--W
    WWWWWW
    </map>
    ```
    **(참고: `<prompt>`와 `<map>` 태그 사이, 그리고 맵 내용과 `</map>` 태그 사이에 줄바꿈이 필요합니다.)**

* (선택 사항) `augment_maps.py`를 사용하여 맵 데이터를 증강합니다.
    ```bash
    # 예시: python augment_maps.py --input_dir maps --output_dir maps_augmented
    ```

* 다음 명령어로 `train_maps.json` 및 `val_maps.json` 파일을 생성합니다:
    ```bash
    python 1_data_preparation.py
    ```

### 3. 모델 및 토크나이저 설정 (Model & Tokenizer Setup)

* 다음 명령어로 DistilGPT2 모델과 커스텀 토큰이 추가된 토크나이저 설정을 저장합니다:
    ```bash
    python 2_model_setup.py
    ```

### 4. 모델 학습 (Training)

* 모델을 파인튜닝하여 `trained_model/` 폴더에 저장합니다:
    ```bash
    python 3_train_model.py
    ```
    
    학습 파라미터는 `3_train_model.py` 스크립트 내에서 필요에 따라 조정 가능합니다.

### 5. 맵 생성 (Generation)

* 새로운 맵을 생성하고, 결과를 콘솔에 출력하고 `generated_maps/` 폴더에 저장합니다:
    ```bash
    python 4_generate_maps.py --prompt "your desired prompt" --count 3 --max_length 150 --temperature 0.7
    ```

* **주요 실행 인자:**
    * `--prompt`: 맵 생성을 위한 텍스트 프롬프트 (예: "wide map, many traps, needs jumping")
    * `--count`: 생성할 맵의 개수 (기본값: 3)
    * `--max_length`: 생성할 텍스트(맵)의 최대 토큰 길이 (기본값: 1024)
    * `--temperature`: 생성 결과의 무작위성 조절 (기본값: 0.7 또는 0.8)
    * `--model_dir`: 학습된 모델 폴더 (기본값: "trained_model")
    * `--output_dir`: 생성된 맵을 저장할 폴더 (기본값: "generated_maps")
    * `--no_prompt`: 프롬프트 없이 생성 시작 (테스트용)

## 주요 참고 논문

* **MarioGPT: Open-Ended Text2Level Generation through Large Language Models**
    * 저자: Sudhakaran, Shyam; González-Duque, Miguel; Glanois, Claire; Freiberger, Matthias; Najarro, Elias; Risi, Sebastian
    * 발행년도: 2023
    * DOI: [10.48550/arxiv.2302.05981](https://doi.org/10.48550/arxiv.2302.05981)
    * arXiv: [2302.05981](https://arxiv.org/abs/2302.05981)
