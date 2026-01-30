"""
Markdown文書をチャンクに分割するスクリプト
CHUNKING_GUIDE.md のルールに従う
"""

import re
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class Chunk:
    """1つのチャンクを表すデータクラス"""
    id: str                    # 一意のID（例: adler_core_001）
    content: str               # チャンク本文
    ancestor: str              # 始祖（Adler, Freud, Jung, Neisser）
    family_line: str           # 学派
    source_author: str         # 著者
    role: str                  # ancestor_core / descendant_extension
    level: str                 # foundation / applied
    chunk_type: str            # premise / core_concept / structure / application / boundary / caution / references
    section_title: str         # セクション名
    source_file: str           # 元ファイル名

def extract_metadata(content: str) -> dict:
    """Markdownファイルからメタデータセクションを抽出"""
    metadata = {}
    meta_match = re.search(r'## メタデータ\n(.*?)\n---', content, re.DOTALL)
    if meta_match:
        meta_text = meta_match.group(1)
        for line in meta_text.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().replace('- ', '')
                metadata[key] = value.strip()
    return metadata

def determine_chunk_type(section_title: str) -> str:
    """セクションタイトルからchunk_typeを決定"""
    title_lower = section_title.lower()
    if '基本前提' in section_title:
        return 'premise'
    elif '中核概念' in section_title:
        return 'core_concept'
    elif '理論構造' in section_title:
        return 'structure'
    elif '力点' in section_title or '説明' in section_title or '接続' in section_title:
        return 'application'
    elif '誤解' in section_title or '境界' in section_title:
        return 'boundary'
    elif '留意' in section_title or '三角コーン' in section_title:
        return 'caution'
    elif '文献' in section_title:
        return 'references'
    else:
        return 'core_concept'  # デフォルト

def split_by_headings(content: str, level: int = 2) -> list[tuple[str, str]]:
    """
    指定レベルの見出しで分割
    Returns: [(section_title, section_content), ...]
    """
    pattern = r'^(#{' + str(level) + r'})\s+(.+)$'
    sections = []
    current_title = ""
    current_content = []
    
    for line in content.split('\n'):
        match = re.match(pattern, line)
        if match:
            if current_title or current_content:
                sections.append((current_title, '\n'.join(current_content)))
            current_title = match.group(2)
            current_content = []
        else:
            current_content.append(line)
    
    if current_title or current_content:
        sections.append((current_title, '\n'.join(current_content)))
    
    return sections

def chunk_document(filepath: Path) -> list[Chunk]:
    """
    1つのMarkdownファイルをチャンクに分割
    """
    content = filepath.read_text(encoding='utf-8')
    filename = filepath.stem
    metadata = extract_metadata(content)
    
    chunks = []
    chunk_counter = 0
    
    # レベル2見出し（##）で大分割
    level2_sections = split_by_headings(content, level=2)
    
    for section_title, section_content in level2_sections:
        # メタデータセクションはスキップ
        if 'メタデータ' in section_title:
            continue
        
        # 「理論の骨格」セクションはレベル3で細分割
        if '理論の骨格' in section_title:
            level3_sections = split_by_headings(section_content, level=3)
            for sub_title, sub_content in level3_sections:
                if not sub_title:
                    continue
                
                # 中核概念がさらに長い場合はレベル4で分割
                if '中核概念' in sub_title and len(sub_content) > 6000:
                    # **（1）見出し** パターンで分割
                    concept_pattern = r'\*\*（\d+）(.+?）\*\*'
                    concept_sections = re.split(concept_pattern, sub_content)
                    
                    # 分割結果を処理
                    i = 0
                    while i < len(concept_sections):
                        if i + 1 < len(concept_sections) and re.match(r'.+', concept_sections[i]):
                            concept_title = concept_sections[i]
                            concept_content = concept_sections[i + 1] if i + 1 < len(concept_sections) else ""
                            
                            chunk_counter += 1
                            full_title = f"{sub_title} - {concept_title}"
                            chunks.append(Chunk(
                                id=f"{filename}_{chunk_counter:03d}",
                                content=f"**（{concept_title}）**\n{concept_content}".strip(),
                                ancestor=metadata.get('ancestor', ''),
                                family_line=metadata.get('family_line', ''),
                                source_author=metadata.get('source_author', ''),
                                role=metadata.get('role', ''),
                                level=metadata.get('level', ''),
                                chunk_type=determine_chunk_type(full_title),
                                section_title=full_title,
                                source_file=filename
                            ))
                            i += 2
                        else:
                            i += 1
                else:
                    # 通常のレベル3セクション
                    chunk_counter += 1
                    chunks.append(Chunk(
                        id=f"{filename}_{chunk_counter:03d}",
                        content=sub_content.strip(),
                        ancestor=metadata.get('ancestor', ''),
                        family_line=metadata.get('family_line', ''),
                        source_author=metadata.get('source_author', ''),
                        role=metadata.get('role', ''),
                        level=metadata.get('level', ''),
                        chunk_type=determine_chunk_type(sub_title),
                        section_title=sub_title,
                        source_file=filename
                    ))
        else:
            # 他のレベル2セクション（力点、誤解、留意点など）
            # レベル3があれば分割、なければそのまま
            level3_sections = split_by_headings(section_content, level=3)
            
            if len(level3_sections) > 1:
                for sub_title, sub_content in level3_sections:
                    if not sub_title and not sub_content.strip():
                        continue
                    chunk_counter += 1
                    full_title = f"{section_title} - {sub_title}" if sub_title else section_title
                    chunks.append(Chunk(
                        id=f"{filename}_{chunk_counter:03d}",
                        content=sub_content.strip(),
                        ancestor=metadata.get('ancestor', ''),
                        family_line=metadata.get('family_line', ''),
                        source_author=metadata.get('source_author', ''),
                        role=metadata.get('role', ''),
                        level=metadata.get('level', ''),
                        chunk_type=determine_chunk_type(full_title),
                        section_title=full_title,
                        source_file=filename
                    ))
            else:
                chunk_counter += 1
                chunks.append(Chunk(
                    id=f"{filename}_{chunk_counter:03d}",
                    content=section_content.strip(),
                    ancestor=metadata.get('ancestor', ''),
                    family_line=metadata.get('family_line', ''),
                    source_author=metadata.get('source_author', ''),
                    role=metadata.get('role', ''),
                    level=metadata.get('level', ''),
                    chunk_type=determine_chunk_type(section_title),
                    section_title=section_title,
                    source_file=filename
                ))
    
    return chunks

def process_all_documents(input_dir: Path, output_path: Path):
    """全文書を処理してJSONに出力"""
    all_chunks = []
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} not found.")
        return

    # 再帰的に検索するか、documents/rag_documentsなど構成に合わせて調整
    # ここでは input_dir 直下を見る想定
    files = list(input_dir.glob('*.md'))
    if not files:
        print(f"Warning: No markdown files found in {input_dir}")
        
    for filepath in sorted(files):
        if filepath.name.startswith('CHUNKING') or filepath.name.startswith('RAG_DESIGN'): 
            continue
        print(f"Processing: {filepath.name}")
        try:
            chunks = chunk_document(filepath)
            all_chunks.extend(chunks)
            print(f"  -> {len(chunks)} chunks")
        except Exception as e:
            print(f"  -> Error processing {filepath.name}: {e}")
    
    # JSON出力
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = [asdict(chunk) for chunk in all_chunks]
    output_path.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\nTotal: {len(all_chunks)} chunks -> {output_path}")

if __name__ == "__main__":
    # パス調整: スクリプト実行位置からの相対パス
    # プロジェクトルートから実行することを想定
    input_dir = Path("documents/rag_documents")
    output_path = Path("data/chunks.json")
    
    print(f"Search path: {input_dir.absolute()}")
    process_all_documents(input_dir, output_path)
