"""
SciTSR 데이터셋을 SEMv3 LRC 형식으로 변환

SciTSR 데이터셋 구조:
- structure/*.json: 셀 정보 (cells with start_row, end_row, start_col, end_col)
- chunk/*.chunk: 텍스트 청크 정보 (JSON 형식)
- rel/*.rel: 관계 정보 (선택사항)
- img/*.png: 이미지 파일

사용법:
    python libs/convert2lrc.py \
        --scitsr_dir ./data/SciTSR \
        --split train \
        --output_dir ./data/lrc \
        --output_name train_v5
"""

import os
import sys
import json
import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# SEMv3 라이브러리 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libs.data.list_record_cache import ListRecordCacher


def load_chunk_file(chunk_path: str) -> Optional[List[Dict]]:
    """Chunk 파일 로드 (JSON 형식)"""
    try:
        with open(chunk_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('chunks', [])
    except Exception as e:
        print(f"Warning: Failed to load chunk file {chunk_path}: {e}")
        return None


def load_structure_file(structure_path: str) -> Optional[Dict]:
    """Structure JSON 파일 로드"""
    try:
        with open(structure_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load structure file {structure_path}: {e}")
        return None


def create_layout_from_cells(cells: List[Dict], n_rows: int, n_cols: int) -> np.ndarray:
    """
    Cells 정보로부터 layout 배열 생성
    
    Args:
        cells: 셀 리스트, 각 셀은 start_row, end_row, start_col, end_col 포함
        n_rows: 테이블 행 수
        n_cols: 테이블 열 수
    
    Returns:
        layout: (n_rows, n_cols) numpy array
    """
    layout = np.zeros((n_rows, n_cols), dtype=np.int32)
    
    # 각 셀에 대해 layout 배열 채우기
    for cell in cells:
        cell_id = cell.get('id', 0)
        start_row = cell.get('start_row', 0)
        end_row = cell.get('end_row', 0)
        start_col = cell.get('start_col', 0)
        end_col = cell.get('end_col', 0)
        
        # 범위 확인 및 조정
        start_row = max(0, min(start_row, n_rows - 1))
        end_row = max(start_row, min(end_row, n_rows - 1))
        start_col = max(0, min(start_col, n_cols - 1))
        end_col = max(start_col, min(end_col, n_cols - 1))
        
        # 해당 영역에 셀 ID 할당
        layout[start_row:end_row+1, start_col:end_col+1] = cell_id
    
    return layout


def infer_table_size(cells: List[Dict]) -> Tuple[int, int]:
    """
    Cells 정보로부터 테이블 크기 추론
    
    Returns:
        (n_rows, n_cols): 테이블의 행 수와 열 수
    """
    max_row = 0
    max_col = 0
    
    for cell in cells:
        end_row = cell.get('end_row', cell.get('start_row', 0))
        end_col = cell.get('end_col', cell.get('start_col', 0))
        max_row = max(max_row, end_row)
        max_col = max(max_col, end_col)
    
    # 0-based 인덱스이므로 +1
    n_rows = max_row + 1
    n_cols = max_col + 1
    
    return n_rows, n_cols


def extract_table_features(
    chunks: Optional[List[Dict]],
    cells: List[Dict],
    n_rows: int,
    n_cols: int,
    img_w: int,
    img_h: int
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Chunk와 Cell 정보로부터 테이블 feature 추출
    
    Returns:
        row_split_lines: List of row line coordinates
        col_split_lines: List of col line coordinates
        row_start_center_bboxes: (n_rows, 4) array
        col_start_center_bboxes: (n_cols, 4) array
        row_line_segmentations: List of row line polygons
        col_line_segmentations: List of col line polygons
        row_line_points: List of row line points
        col_line_points: List of col line points
    """
    
    # Cell ID별로 bbox 수집
    cell_bboxes = {}  # {cell_id: [[x0, y0, x1, y1], ...]}
    
    if chunks:
        for chunk in chunks:
            pos = chunk.get('pos', [])
            text = chunk.get('text', '').strip()
            
            if len(pos) >= 4:
                x0, y0, x1, y1 = pos[:4]
                
                # 이 chunk가 어느 cell에 속하는지 찾기
                for cell in cells:
                    cell_id = cell['id']
                    cell_content = ' '.join(cell.get('content', [])).strip()
                    
                    # Text 매칭 (공백 제거하고 비교)
                    text_normalized = text.replace(' ', '').lower()
                    content_normalized = cell_content.replace(' ', '').lower()
                    
                    if text_normalized in content_normalized or content_normalized in text_normalized:
                        if cell_id not in cell_bboxes:
                            cell_bboxes[cell_id] = []
                        cell_bboxes[cell_id].append([x0, y0, x1, y1])
                        break
    
    # Row/Col별 좌표 수집
    row_coords = {}  # {row_idx: {'min_y': [], 'max_y': []}}
    col_coords = {}  # {col_idx: {'min_x': [], 'max_x': []}}
    
    for cell in cells:
        cell_id = cell['id']
        start_row = cell['start_row']
        end_row = cell['end_row']
        start_col = cell['start_col']
        end_col = cell['end_col']
        
        if cell_id in cell_bboxes:
            # 이 cell의 모든 bbox 통합
            all_x0 = [bbox[0] for bbox in cell_bboxes[cell_id]]
            all_y0 = [bbox[1] for bbox in cell_bboxes[cell_id]]
            all_x1 = [bbox[2] for bbox in cell_bboxes[cell_id]]
            all_y1 = [bbox[3] for bbox in cell_bboxes[cell_id]]
            
            cell_x0 = min(all_x0)
            cell_y0 = min(all_y0)
            cell_x1 = max(all_x1)
            cell_y1 = max(all_y1)
            
            # Row 좌표 저장
            for r in range(start_row, end_row + 1):
                if r not in row_coords:
                    row_coords[r] = {'min_y': [], 'max_y': []}
                row_coords[r]['min_y'].append(cell_y0)
                row_coords[r]['max_y'].append(cell_y1)
            
            # Col 좌표 저장
            for c in range(start_col, end_col + 1):
                if c not in col_coords:
                    col_coords[c] = {'min_x': [], 'max_x': []}
                col_coords[c]['min_x'].append(cell_x0)
                col_coords[c]['max_x'].append(cell_x1)
    
    # Row boundaries 계산 - 정확히 n_rows+1개
    row_boundaries = []
    row_boundaries.append(0.0)  # 첫 번째
    
    for i in range(n_rows):
        if i in row_coords and (i + 1) in row_coords:
            # 현재 row의 max와 다음 row의 min의 중간
            y_max = np.mean(row_coords[i]['max_y'])
            y_min_next = np.mean(row_coords[i + 1]['min_y'])
            boundary = (y_max + y_min_next) / 2
        elif i in row_coords:
            boundary = np.mean(row_coords[i]['max_y'])
        else:
            # Fallback: 균등 분할
            boundary = float((i + 1) * img_h / n_rows)
        
        if i < n_rows - 1:  # 마지막은 제외
            row_boundaries.append(float(boundary))
    
    row_boundaries.append(float(img_h))  # 마지막
    
    # 정확히 n_rows+1개가 되도록 보장
    assert len(row_boundaries) == n_rows + 1, f"Expected {n_rows+1} row boundaries, got {len(row_boundaries)}"
    
    # Col boundaries 계산 - 정확히 n_cols+1개
    col_boundaries = []
    col_boundaries.append(0.0)
    
    for i in range(n_cols):
        if i in col_coords and (i + 1) in col_coords:
            x_max = np.mean(col_coords[i]['max_x'])
            x_min_next = np.mean(col_coords[i + 1]['min_x'])
            boundary = (x_max + x_min_next) / 2
        elif i in col_coords:
            boundary = np.mean(col_coords[i]['max_x'])
        else:
            boundary = float((i + 1) * img_w / n_cols)
        
        if i < n_cols - 1:
            col_boundaries.append(float(boundary))
    
    col_boundaries.append(float(img_w))
    
    # 정확히 n_cols+1개가 되도록 보장
    assert len(col_boundaries) == n_cols + 1, f"Expected {n_cols+1} col boundaries, got {len(col_boundaries)}"
    
    # 1. Row split lines (horizontal lines)
    row_split_lines = []
    for y in row_boundaries:
        row_split_lines.append(np.array([[0.0, y], [float(img_w), y]], dtype=np.float32))
    
    # 2. Col split lines (vertical lines)
    col_split_lines = []
    for x in col_boundaries:
        col_split_lines.append(np.array([[x, 0.0], [x, float(img_h)]], dtype=np.float32))
    
    # 3. Row start center bboxes - 각 row의 영역 (n_rows개)
    row_start_center_bboxes = []
    for i in range(n_rows):
        y1 = row_boundaries[i]
        y2 = row_boundaries[i + 1]
        # 왼쪽 가장자리에 작은 bbox
        bbox = np.array([0.0, y1, 50.0, y2], dtype=np.float32)
        row_start_center_bboxes.append(bbox)
    
    if len(row_start_center_bboxes) == 0:
        row_start_center_bboxes = np.zeros((0, 4), dtype=np.float32)
    else:
        row_start_center_bboxes = np.array(row_start_center_bboxes, dtype=np.float32)
    
    # 4. Col start center bboxes - 각 col의 영역 (n_cols개)
    col_start_center_bboxes = []
    for i in range(n_cols):
        x1 = col_boundaries[i]
        x2 = col_boundaries[i + 1]
        # 상단 가장자리에 작은 bbox
        bbox = np.array([x1, 0.0, x2, 50.0], dtype=np.float32)
        col_start_center_bboxes.append(bbox)
    
    if len(col_start_center_bboxes) == 0:
        col_start_center_bboxes = np.zeros((0, 4), dtype=np.float32)
    else:
        col_start_center_bboxes = np.array(col_start_center_bboxes, dtype=np.float32)
    
    # 5 & 6. Line segmentations - 내부 경계선만 (첫/마지막 제외)
    row_line_segmentations = []
    for i in range(n_rows):
        y = row_boundaries[i + 1]  # 각 row의 하단 경계
        row_line_segmentations.append(np.array([[0.0, y], [float(img_w), y]], dtype=np.float32))

    col_line_segmentations = []
    for i in range(n_cols):
        x = col_boundaries[i + 1]  # 각 col의 우측 경계
        col_line_segmentations.append(np.array([[x, 0.0], [x, float(img_h)]], dtype=np.float32))

    # 7 & 8. Line points - segmentations와 동일
    row_line_points = [line.copy() for line in row_line_segmentations]
    col_line_points = [line.copy() for line in col_line_segmentations]

    return (row_split_lines, col_split_lines,
        row_start_center_bboxes, col_start_center_bboxes,
        row_line_segmentations, col_line_segmentations,
        row_line_points, col_line_points)


def process_scitsr_table(
    base_name: str,
    scitsr_dir: str,
    image_dir: Optional[str] = None
) -> Optional[Dict]:
    """
    SciTSR 테이블 데이터를 SEMv3 LRC 레코드로 변환
    
    Args:
        base_name: 파일 기본 이름 (확장자 제외, 예: "0704.2596v1.2")
        scitsr_dir: SciTSR 데이터셋 디렉토리
        image_dir: 이미지 디렉토리 (None이면 scitsr_dir/img 사용)
    
    Returns:
        record: SEMv3 LRC 레코드 딕셔너리 또는 None
    """
    if image_dir is None:
        image_dir = os.path.join(scitsr_dir, 'img')
    
    # 파일 경로
    structure_path = os.path.join(scitsr_dir, 'structure', f'{base_name}.json')
    chunk_path = os.path.join(scitsr_dir, 'chunk', f'{base_name}.chunk')
    image_path = os.path.join(image_dir, f'{base_name}.png')
    
    # Structure 파일 로드
    structure_data = load_structure_file(structure_path)
    if structure_data is None:
        return None
    
    cells = structure_data.get('cells', [])
    if not cells:
        print(f"Warning: No cells found in {base_name}")
        return None
    
    # 이미지 확인
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return None
    
    # 이미지 크기 읽기
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Failed to load image: {image_path}")
        return None
    
    h, w = img.shape[:2]
    
    # 테이블 크기 추론
    n_rows, n_cols = infer_table_size(cells)
    
    # Layout 생성
    layout = create_layout_from_cells(cells, n_rows, n_cols)
    
    # Chunk 정보 로드
    chunks = load_chunk_file(chunk_path)
    
    # 테이블 feature 추출
    (row_split_lines, col_split_lines,
     row_start_center_bboxes, col_start_center_bboxes,
     row_line_segmentations, col_line_segmentations,
     row_line_points, col_line_points) = extract_table_features(
        chunks, cells, n_rows, n_cols, w, h
    )
    
    # 레코드 생성 - 모든 필수 필드 포함
    record = {
        'image_path': os.path.abspath(image_path),
        'image_w': w,
        'image_h': h,
        'layout': layout,
        'row_split_lines': row_split_lines,
        'col_split_lines': col_split_lines,
        'row_start_center_bboxes': row_start_center_bboxes,
        'col_start_center_bboxes': col_start_center_bboxes,
        'row_line_segmentations': row_line_segmentations,
        'col_line_segmentations': col_line_segmentations,
        'row_line_points': row_line_points,
        'col_line_points': col_line_points,
        'rota_angle': 0.0, 
    }
    
    return record


def convert_scitsr_to_lrc(
    scitsr_dir: str,
    output_dir: str,
    output_name: str,
    split: str = 'val'
):
    """
    SciTSR 데이터셋을 LRC로 변환
    
    Args:
        scitsr_dir: SciTSR 데이터셋 루트 디렉토리 (예: ./data/SciTSR)
        output_dir: 출력 디렉토리
        output_name: 출력 파일명 (확장자 제외)
        split: 데이터셋 분할 ('train', 'val', 'test')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # SciTSR 데이터셋 경로
    split_dir = os.path.join(scitsr_dir, split)
    structure_dir = os.path.join(split_dir, 'structure')
    image_dir = os.path.join(split_dir, 'img')
    
    if not os.path.exists(structure_dir):
        raise ValueError(f"Structure directory not found: {structure_dir}")
    if not os.path.exists(image_dir):
        raise ValueError(f"Image directory not found: {image_dir}")
    
    # Structure 파일 목록 가져오기
    structure_files = sorted([
        f for f in os.listdir(structure_dir)
        if f.endswith('.json')
    ])
    
    if not structure_files:
        raise ValueError(f"No structure files found in {structure_dir}")
    
    print(f"Found {len(structure_files)} structure files")
    
    # LRC 파일 생성
    lrc_path = os.path.join(output_dir, f"{output_name}.lrc")
    cacher = ListRecordCacher(lrc_path)
    
    # Info 파일 생성
    info_path = os.path.join(output_dir, f"{output_name}.lrc_info.txt")
    info_lines = []
    
    record_count = 0
    failed_count = 0
    
    for structure_file in structure_files:
        try:
            # 기본 이름 추출 (확장자 제외)
            base_name = structure_file.replace('.json', '')
            
            # 레코드 생성
            record = process_scitsr_table(base_name, split_dir, image_dir)
            
            if record is None:
                failed_count += 1
                continue
            
            # LRC에 추가
            cacher.add_record(record)
            
            # Info 파일용 정보 - 실제 이미지 경로 사용
            h = record['image_h']
            w = record['image_w']
            height_ave = h
            info_lines.append(f"{record['image_path']}\t{h}\t{w}\t{height_ave}")
            
            record_count += 1
            
            if record_count % 100 == 0:
                print(f"Processed {record_count} records...")
        
        except Exception as e:
            print(f"Error processing {structure_file}: {e}")
            import traceback
            traceback.print_exc()
            failed_count += 1
            continue
    
    # LRC 파일 완료
    cacher.flush()
    cacher.close()
    
    # Info 파일 작성
    print(f"Writing info file: {info_path}")
    with open(info_path, 'w', encoding='utf-8') as f:
        for line in info_lines:
            f.write(line + '\n')
    
    print(f"\n✅ Done! Created:")
    print(f"  - LRC file: {lrc_path} ({record_count} records)")
    print(f"  - Info file: {info_path}")
    if failed_count > 0:
        print(f"  - Failed: {failed_count} records")
    
    return lrc_path, info_path


def main():
    parser = argparse.ArgumentParser(
        description='Convert SciTSR dataset to SEMv3 LRC format'
    )
    parser.add_argument('--scitsr_dir', type=str, required=True,
                       help='SciTSR dataset root directory (e.g., ./data/SciTSR)')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to convert')
    parser.add_argument('--output_dir', type=str, default='./data/lrc',
                       help='Output directory for LRC files')
    parser.add_argument('--output_name', type=str, default='train_v5',
                       help='Output file name (without extension)')
    
    args = parser.parse_args()
    
    convert_scitsr_to_lrc(
        scitsr_dir=args.scitsr_dir,
        output_dir=args.output_dir,
        output_name=args.output_name,
        split=args.split
    )


if __name__ == '__main__':
    main()