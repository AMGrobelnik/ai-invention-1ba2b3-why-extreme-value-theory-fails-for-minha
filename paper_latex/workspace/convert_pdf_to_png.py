#!/usr/bin/env python3
"""Convert PDF pages to PNG images for visual review."""

import os
import sys

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf

def convert_pdf_to_png(pdf_path, output_dir, dpi=150):
    """Convert each page of a PDF to PNG image."""
    os.makedirs(output_dir, exist_ok=True)
    
    doc = pymupdf.open(pdf_path)
    zoom = dpi / 72  # 72 DPI is the base
    
    page_paths = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        mat = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        output_path = os.path.join(output_dir, f"page_{page_num + 1:02d}.png")
        pix.save(output_path)
        page_paths.append(output_path)
        print(f"Saved page {page_num + 1} to {output_path}")
    
    doc.close()
    return page_paths

if __name__ == "__main__":
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "paper.pdf"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "page_images"
    
    print(f"Converting {pdf_path} to PNG images at 150 DPI...")
    page_paths = convert_pdf_to_png(pdf_path, output_dir, dpi=150)
    print(f"\nConverted {len(page_paths)} pages to {output_dir}/")
