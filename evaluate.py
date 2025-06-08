#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate.py - OCR vs GPT Evaluation Tool

This module provides functions to evaluate OCR accuracy against GPT results:
1. Compare existing CSV results (calculate_similarity_stats)
2. Re-run OCR and compare with GPT results (evaluate_ocr_vs_gpt)

Both functions use identical text comparison methods for consistency.
"""

import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher
from pororo import Pororo
from typing import Dict, Tuple


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison by removing extra whitespace and special characters.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text (lowercase, no special chars except Korean)
    """
    if not text or pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces -> single space
    text = re.sub(r'[^\w\s가-힣]', '', text)  # Remove special chars except Korean
    return text.strip().lower()


def compare_texts(text1: str, text2: str) -> Tuple[bool, bool, float]:
    """
    Compare two texts using multiple criteria.
    
    Args:
        text1: First text to compare
        text2: Second text to compare
        
    Returns:
        Tuple of (exact_match, similar_match, similarity_score)
        - exact_match: True if texts are identical
        - similar_match: True if normalized texts are identical
        - similarity_score: Float between 0.0-1.0 using SequenceMatcher
    """
    # Handle None/NaN cases
    if not text1 or pd.isna(text1):
        text1 = ""
    if not text2 or pd.isna(text2):
        text2 = ""
    
    text1 = str(text1)
    text2 = str(text2)
    
    # 1. Exact match
    exact_match = text1.strip() == text2.strip()
    
    # 2. Normalized match (ignore spacing, special chars)
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    similar_match = norm1 == norm2
    
    # 3. Simple similarity score - if normalized texts are the same, return 1.0
    if similar_match and norm1:  # Don't give 1.0 for empty strings
        similarity = 1.0
    elif not norm1 and not norm2:  # Both empty
        similarity = 1.0
    else:
        # Use SequenceMatcher only if they're different
        similarity = SequenceMatcher(None, norm1, norm2).ratio()
    
    return exact_match, similar_match, similarity


def calculate_similarity_stats(csv_path: str, exclude_handwriting: bool = False) -> Dict:
    """
    Calculate similarity statistics from existing CSV results (NaN excluded).
    
    Args:
        csv_path: Path to CSV file containing OCR vs GPT results
        exclude_handwriting: If True, exclude handwriting samples from analysis
        
    Returns:
        Dictionary containing accuracy statistics
    """
    print(f"Reading CSV: {csv_path}")
    
    # Read CSV file
    df = pd.read_csv(csv_path, keep_default_na=False)
    print(f"Total rows in CSV: {len(df)}")
    
    # Debug: Check image path patterns
    print("\nDEBUG: Image path analysis")
    path_patterns = df['image_path'].str.extract(r'(handwriting|image|notice)', expand=False).value_counts(dropna=False)
    print(f"Path patterns: {path_patterns}")
    
    # Show sample paths
    print(f"Sample paths:")
    for i, path in enumerate(df['image_path'].unique()[:5]):
        print(f"  {path}")
    
    # Filter out invalid data
    valid_mask = ~df['gpt_text'].isin(['nan', 'NaN', '', 'null', 'None'])
    valid_mask = valid_mask & df['gpt_text'].notna()
    
    # Convert similarity_score to numeric
    df['similarity_score'] = pd.to_numeric(df['similarity_score'], errors='coerce')
    valid_mask = valid_mask & df['similarity_score'].notna()
    
    print(f"After basic filtering: {valid_mask.sum()} rows")
    
    # Optionally exclude handwriting samples
    if exclude_handwriting:
        # Try different patterns
        handwriting_mask1 = df['image_path'].str.contains('handwriting/', na=False)
        handwriting_mask2 = df['image_path'].str.contains('handwriting', na=False)
        
        print(f"Handwriting pattern 'handwriting/': {handwriting_mask1.sum()} matches")
        print(f"Handwriting pattern 'handwriting': {handwriting_mask2.sum()} matches")
        
        # Use the pattern that actually matches
        if handwriting_mask2.sum() > handwriting_mask1.sum():
            valid_mask = valid_mask & ~handwriting_mask2
            print("Using 'handwriting' pattern (without slash)")
        else:
            valid_mask = valid_mask & ~handwriting_mask1
            print("Using 'handwriting/' pattern (with slash)")
        
        print("Excluding handwriting samples from analysis")
    
    # Always exclude handwriting/B samples (if you want this permanently)
    handwriting_b_mask = df['image_path'].str.contains('handwriting/B', na=False)
    print(f"Handwriting/B samples found: {handwriting_b_mask.sum()}")
    valid_mask = valid_mask & ~handwriting_b_mask
    print("Excluding handwriting/B samples from analysis")
    
    # Apply filters
    valid_df = df[valid_mask]
    print(f"Valid rows (after all filters): {len(valid_df)}")
    print(f"Excluded rows: {len(df) - len(valid_df)}")
    
    if len(valid_df) == 0:
        print("No valid data found!")
        return {"error": "No valid data"}
    
    # Calculate statistics
    total_comparisons = len(valid_df)
    exact_matches = valid_df['exact_match'].sum()
    similar_matches = valid_df['similar_match'].sum()
    similarity_scores = valid_df['similarity_score']
    
    # Basic statistics
    avg_similarity = similarity_scores.mean()
    min_similarity = similarity_scores.min()
    max_similarity = similarity_scores.max()
    std_similarity = similarity_scores.std()
    
    # Accuracy calculations
    exact_accuracy = (exact_matches / total_comparisons) * 100
    similar_accuracy = (similar_matches / total_comparisons) * 100
    high_similarity_count = (similarity_scores >= 0.8).sum()
    high_similarity_percent = (high_similarity_count / total_comparisons) * 100
    
    # Print results
    print(f"\n{'='*60}")
    print("Similarity Statistics (NaN excluded)")
    print(f"{'='*60}")
    print(f"Total valid comparisons: {total_comparisons:,}")
    print(f"\nAccuracy Metrics:")
    print(f"  Exact matches: {exact_matches:,} ({exact_accuracy:.2f}%)")
    print(f"  Similar matches: {similar_matches:,} ({similar_accuracy:.2f}%)")
    print(f"  High similarity (≥0.8): {high_similarity_count:,} ({high_similarity_percent:.2f}%)")
    print(f"\nSimilarity Score Statistics:")
    print(f"  Average: {avg_similarity:.6f}")
    print(f"  Minimum: {min_similarity:.6f}")
    print(f"  Maximum: {max_similarity:.6f}")
    print(f"  Standard Deviation: {std_similarity:.6f}")
    
    # Distribution analysis
    print(f"\nSimilarity Score Distribution:")
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for i in range(len(bins)-1):
        if i == len(bins)-2:  # Last bin includes 1.0
            count = ((similarity_scores >= bins[i]) & (similarity_scores <= bins[i+1])).sum()
        else:
            count = ((similarity_scores >= bins[i]) & (similarity_scores < bins[i+1])).sum()
        percent = (count / total_comparisons) * 100
        print(f"  {bins[i]:.1f} - {bins[i+1]:.1f}: {count:,} ({percent:.1f}%)")
    
    # Show worst performing samples
    print(f"\nLowest Similarity Samples (top 5):")
    lowest_sim = valid_df.nsmallest(5, 'similarity_score')[
        ['image_path', 'bbox_index', 'ocr_text', 'gpt_text', 'similarity_score']
    ]
    for _, row in lowest_sim.iterrows():
        print(f"  {row['similarity_score']:.3f}: OCR='{row['ocr_text'][:30]}...' | "
              f"GPT='{row['gpt_text'][:30]}...'")
    print(f"{'='*60}")
    
    return {
        'total_comparisons': total_comparisons,
        'exact_accuracy': exact_accuracy,
        'similar_accuracy': similar_accuracy,
        'average_similarity': avg_similarity,
        'high_similarity_percent': high_similarity_percent,
        'min_similarity': min_similarity,
        'max_similarity': max_similarity,
        'std_similarity': std_similarity
    }


def evaluate_ocr_vs_gpt(csv_path: str, exclude_handwriting: bool = False) -> Dict:
    """
    Re-run OCR on images and compare results with GPT predictions from CSV.
    
    Args:
        csv_path: Path to CSV file containing GPT results
        exclude_handwriting: If True, exclude handwriting samples from analysis
        
    Returns:
        Dictionary containing accuracy statistics
    """
    print("Loading OCR model...")
    ocr = Pororo(task="ocr", lang="ko", model="brainocr")
    
    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path, keep_default_na=False)
    print(f"Total rows in CSV: {len(df)}")
    
    # Optionally exclude handwriting samples
    if exclude_handwriting:
        df = df[~df['image_path'].str.contains('handwriting/', na=False)]
        print(f"After excluding handwriting: {len(df)} rows")
    
    # Group by image path
    image_groups = df.groupby('image_path')
    print(f"Processing {len(image_groups)} unique images")
    
    total_exact = 0
    total_similar = 0
    total_similarity = 0.0
    total_count = 0
    skipped_ocr_failed = 0
    skipped_nan_gpt = 0
    skipped_bbox_mismatch = 0
    
    for image_path, group in image_groups:
        print(f"Processing: {image_path} ({len(group)} bboxes)")
        
        # Run OCR on image
        try:
            ocr_result = ocr(image_path, detail=True)
            if not ocr_result.get('bounding_poly'):
                print(f"  No text detected in image")
                skipped_ocr_failed += len(group)
                continue
        except Exception as e:
            print(f"  OCR failed: {e}")
            skipped_ocr_failed += len(group)
            continue
        
        print(f"  OCR detected {len(ocr_result['bounding_poly'])} bboxes")
        
        # Compare each bounding box
        for _, row in group.iterrows():
            bbox_index = int(row['bbox_index'])
            gpt_text = str(row['gpt_text'])
            
            # Skip invalid GPT results
            if gpt_text in ['nan', 'NaN', ''] or pd.isna(row['gpt_text']):
                print(f"    Bbox {bbox_index}: Skipping - GPT text is invalid")
                skipped_nan_gpt += 1
                continue
            
            # Get OCR result for corresponding bbox
            if bbox_index < len(ocr_result['bounding_poly']):
                new_ocr_text = ocr_result['bounding_poly'][bbox_index]['description']
                
                # Compare texts using same method as calculate_similarity_stats
                exact, similar, similarity = compare_texts(new_ocr_text, gpt_text)
                
                total_exact += int(exact)
                total_similar += int(similar)
                total_similarity += similarity
                total_count += 1
                
                print(f"    Bbox {bbox_index}: OCR='{new_ocr_text[:20]}...' | "
                      f"GPT='{gpt_text[:20]}...' | Score={similarity:.3f}")
            else:
                print(f"    Bbox {bbox_index}: Index out of range (OCR has {len(ocr_result['bounding_poly'])} bboxes)")
                skipped_bbox_mismatch += 1
    
    # Print debugging info
    print(f"\nDEBUG INFO:")
    print(f"  Original CSV rows: {len(pd.read_csv(csv_path, keep_default_na=False))}")
    print(f"  After filtering: {len(df)}")
    print(f"  Skipped (OCR failed): {skipped_ocr_failed}")
    print(f"  Skipped (NaN GPT): {skipped_nan_gpt}")
    print(f"  Skipped (bbox mismatch): {skipped_bbox_mismatch}")
    print(f"  Successfully processed: {total_count}")
    
    # Calculate final statistics
    if total_count == 0:
        print("No valid comparisons found!")
        return {"error": "No valid comparisons found"}
    
    exact_accuracy = (total_exact / total_count) * 100
    similar_accuracy = (total_similar / total_count) * 100
    avg_similarity = total_similarity / total_count
    
    # Print results
    print(f"\n{'='*60}")
    print("OCR vs GPT Comparison Results (Fresh OCR)")
    print(f"{'='*60}")
    print(f"Total valid comparisons: {total_count:,}")
    print(f"\nAccuracy Metrics:")
    print(f"  Exact matches: {total_exact:,} ({exact_accuracy:.2f}%)")
    print(f"  Similar matches: {total_similar:,} ({similar_accuracy:.2f}%)")
    print(f"  Average similarity: {avg_similarity:.6f}")
    print(f"{'='*60}")
    
    return {
        'total_comparisons': total_count,
        'exact_matches': total_exact,
        'similar_matches': total_similar,
        'exact_accuracy': exact_accuracy,
        'similar_accuracy': similar_accuracy,
        'average_similarity': avg_similarity
    }

def main():
    
    csv_files = [
        "train_2_ocr_gpt_results_org.csv",
        "test_2_ocr_gpt_results_org.csv"
    ]

    
    for csv_file in csv_files:
        try:
            print(f"\n{'='*60}")
            print(f"Processing: {csv_file}")
            print(f"{'='*60}")
            
            # Method 1: Analyze existing CSV results
            print("\n[Method 1] Analyzing existing CSV results...")
            # stats1 = calculate_similarity_stats(csv_file, exclude_handwriting=False)
            
            # Method 2: Re-run OCR and compare (uncomment if needed)
            print("\n[Method 2] Re-running OCR and comparing...")
            stats2 = evaluate_ocr_vs_gpt(csv_file, exclude_handwriting=False)
            
        except FileNotFoundError:
            print(f"File not found: {csv_file}")
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    
    print(f"\n{'='*80}")
    print("Evaluation completed!")


if __name__ == "__main__":
    main()