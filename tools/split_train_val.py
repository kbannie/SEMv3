"""
SciTSR 원본 데이터셋을 Train과 Validation으로 분할

원본 데이터(structure, chunk, img 파일들)를 train과 val 폴더로 나눕니다.
나중에 각각을 LRC로 변환할 수 있습니다.

사용법:
    python tools/split_train_val.py \
        --scitsr_dir ./data/SciTSR \
        --split_dir train \
        --val_size 1500 \
        --shuffle
"""

import os
import sys
import shutil
import argparse
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def split_scitsr_dataset(
    scitsr_dir: str,
    split_dir: str = 'train',
    val_size: int = 1500,
    shuffle: bool = True,
    seed: int = 42
):
    """
    SciTSR 데이터셋을 Train과 Validation으로 분할
    
    Args:
        scitsr_dir: SciTSR 데이터셋 루트 디렉토리
        split_dir: 분할할 데이터셋 디렉토리 (예: 'train')
        val_size: Validation 데이터셋 크기
        shuffle: 랜덤 셔플 여부
        seed: 랜덤 시드
    """
    # 경로 설정
    source_dir = os.path.join(scitsr_dir, split_dir)
    train_dir = os.path.join(scitsr_dir, 'train')
    val_dir = os.path.join(scitsr_dir, 'val')
    
    # 디렉토리 확인
    if not os.path.exists(source_dir):
        raise ValueError(f"Source directory not found: {source_dir}")
    
    structure_dir = os.path.join(source_dir, 'structure')
    chunk_dir = os.path.join(source_dir, 'chunk')
    img_dir = os.path.join(source_dir, 'img')
    
    if not os.path.exists(structure_dir):
        raise ValueError(f"Structure directory not found: {structure_dir}")
    
    # 원본 train 디렉토리와 대상 train 디렉토리가 같은 경우 처리
    if source_dir == train_dir:
        # 원본을 임시 디렉토리로 이동
        temp_dir = os.path.join(scitsr_dir, f'{split_dir}_original')
        if os.path.exists(temp_dir):
            raise ValueError(f"Temporary directory already exists: {temp_dir}. Please remove it first.")
        print(f"⚠️  Source and target directories are the same. Moving source to temporary directory...")
        shutil.move(source_dir, temp_dir)
        source_dir = temp_dir
        structure_dir = os.path.join(source_dir, 'structure')
        chunk_dir = os.path.join(source_dir, 'chunk')
        img_dir = os.path.join(source_dir, 'img')
    
    # Structure 파일 목록 가져오기
    structure_files = sorted([
        f for f in os.listdir(structure_dir)
        if f.endswith('.json')
    ])
    
    if not structure_files:
        raise ValueError(f"No structure files found in {structure_dir}")
    
    total_files = len(structure_files)
    print(f"📁 Found {total_files} files in {source_dir}")
    
    if val_size >= total_files:
        raise ValueError(f"Validation size ({val_size}) must be less than total files ({total_files})")
    
    # 인덱스 생성
    indices = list(range(total_files))
    
    if shuffle:
        print(f"🔀 Shuffling with seed {seed}...")
        random.seed(seed)
        random.shuffle(indices)
    
    # Train과 Validation 인덱스 분할
    val_indices = set(indices[:val_size])
    train_indices = indices[val_size:]
    
    print(f"📊 Split:")
    print(f"   Train: {len(train_indices)} files")
    print(f"   Validation: {len(val_indices)} files")
    
    # 출력 디렉토리 생성
    for target_dir in [train_dir, val_dir]:
        for subdir in ['structure', 'chunk', 'img']:
            os.makedirs(os.path.join(target_dir, subdir), exist_ok=True)
    
    # 파일 이동 (복사가 아닌 이동)
    print(f"\n📝 Moving files...")
    
    train_count = 0
    val_count = 0
    
    for idx, structure_file in enumerate(structure_files):
        # 기본 이름 추출 (확장자 제외)
        base_name = structure_file.replace('.json', '')
        
        # 파일 경로
        structure_src = os.path.join(structure_dir, structure_file)
        chunk_src = os.path.join(chunk_dir, f'{base_name}.chunk')
        img_src = os.path.join(img_dir, f'{base_name}.png')
        
        # 대상 디렉토리 결정
        if idx in val_indices:
            target_base = val_dir
            val_count += 1
        else:
            target_base = train_dir
            train_count += 1
        
        # 파일 이동
        # Structure 파일
        structure_dst = os.path.join(target_base, 'structure', structure_file)
        shutil.move(structure_src, structure_dst)
        
        # Chunk 파일 (있는 경우)
        if os.path.exists(chunk_src):
            chunk_dst = os.path.join(target_base, 'chunk', f'{base_name}.chunk')
            shutil.move(chunk_src, chunk_dst)
        
        # 이미지 파일 (있는 경우)
        if os.path.exists(img_src):
            img_dst = os.path.join(target_base, 'img', f'{base_name}.png')
            shutil.move(img_src, img_dst)
        
        # 진행 상황 출력
        if (train_count + val_count) % 100 == 0:
            print(f"   Processed {train_count + val_count}/{total_files} files...")
    
    # 임시 디렉토리 정리 (비어있으면 삭제)
    if source_dir != os.path.join(scitsr_dir, split_dir):
        try:
            # 디렉토리가 비어있는지 확인
            if not os.listdir(structure_dir):
                os.rmdir(structure_dir)
            if os.path.exists(chunk_dir) and not os.listdir(chunk_dir):
                os.rmdir(chunk_dir)
            if os.path.exists(img_dir) and not os.listdir(img_dir):
                os.rmdir(img_dir)
            if not os.listdir(source_dir):
                os.rmdir(source_dir)
                print(f"✅ Removed temporary directory: {source_dir}")
        except:
            pass
    
    print(f"\n✅ Done!")
    print(f"   Train: {train_dir} ({train_count} files)")
    print(f"   Validation: {val_dir} ({val_count} files)")
    
    print(f"\n📋 Next steps:")
    print(f"   1. Convert train dataset to LRC:")
    print(f"      python tools/convert_scitsr_to_lrc.py --scitsr_dir {scitsr_dir} --split train --output_dir ./data/lrc --output_name train_v5")
    print(f"   2. Convert validation dataset to LRC:")
    print(f"      python tools/convert_scitsr_to_lrc.py --scitsr_dir {scitsr_dir} --split val --output_dir ./data/lrc --output_name valid_v5")
    
    return train_dir, val_dir


def main():
    parser = argparse.ArgumentParser(
        description='Split SciTSR dataset into train and validation'
    )
    parser.add_argument('--scitsr_dir', type=str, required=True,
                       help='SciTSR dataset root directory')
    parser.add_argument('--split_dir', type=str, default='train',
                       help='Source directory to split (default: train)')
    parser.add_argument('--val_size', type=int, default=1500,
                       help='Number of files for validation dataset')
    parser.add_argument('--shuffle', action='store_true',
                       help='Shuffle files before splitting')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for shuffling')
    
    args = parser.parse_args()
    
    split_scitsr_dataset(
        scitsr_dir=args.scitsr_dir,
        split_dir=args.split_dir,
        val_size=args.val_size,
        shuffle=args.shuffle,
        seed=args.seed
    )


if __name__ == '__main__':
    main()

