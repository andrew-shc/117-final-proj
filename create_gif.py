#!/usr/bin/env python3
"""
Create GIFs from image sequences in the project directories.
"""

import os
from PIL import Image
import argparse
from pathlib import Path


def create_gif_from_frames(input_dir, output_path, fps=30, max_frames=None, skip_frames=1):
    """
    Create a GIF from a sequence of image frames.
    
    Args:
        input_dir: Directory containing the frame images
        output_path: Path for the output GIF file
        fps: Frames per second for the GIF
        max_frames: Maximum number of frames to include (None for all)
        skip_frames: Only include every Nth frame (1 means include all frames)
    """
    # Get all image files sorted by name
    frame_files = sorted([f for f in os.listdir(input_dir) 
                         if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    if not frame_files:
        print(f"No image files found in {input_dir}")
        return
    
    # Apply frame skipping
    if skip_frames > 1:
        frame_files = frame_files[::skip_frames]
    
    # Limit number of frames
    if max_frames:
        frame_files = frame_files[:max_frames]
    
    print(f"Loading {len(frame_files)} frames from {input_dir}...")
    
    # Load all frames
    frames = []
    for i, frame_file in enumerate(frame_files):
        if i % 50 == 0:
            print(f"  Loading frame {i+1}/{len(frame_files)}...")
        
        frame_path = os.path.join(input_dir, frame_file)
        img = Image.open(frame_path)
        
        # Convert to RGB if needed (some PNGs might be RGBA)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        frames.append(img)
    
    # Calculate duration per frame in milliseconds
    duration = int(1000 / fps)
    
    # Save as GIF
    print(f"Saving GIF to {output_path}...")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=False
    )
    
    # Get file size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ GIF created successfully: {output_path}")
    print(f"  Frames: {len(frames)}, FPS: {fps}, Size: {size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='Create GIFs from image sequences')
    parser.add_argument('input_dir', nargs='?', help='Input directory containing frames')
    parser.add_argument('-o', '--output', help='Output GIF path')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second (default: 30)')
    parser.add_argument('--max-frames', type=int, help='Maximum number of frames to include')
    parser.add_argument('--skip', type=int, default=1, help='Include every Nth frame (default: 1)')
    parser.add_argument('--all', action='store_true', help='Create GIFs for all frame directories')
    
    args = parser.parse_args()
    
    project_dir = Path(__file__).parent
    
    if args.all:
        # Create GIFs for all common frame directories
        frame_dirs = [
            ('data/IMG_9182_frames', 'outputs/IMG_9182_frames.gif'),
            ('data/IMG_9184_frames', 'outputs/IMG_9184_frames.gif'),
            ('outputs/renders', 'outputs/renders.gif'),
        ]
        
        for input_dir, output_path in frame_dirs:
            full_input = project_dir / input_dir
            full_output = project_dir / output_path
            
            if full_input.exists():
                print(f"\n{'='*60}")
                print(f"Processing: {input_dir}")
                print(f"{'='*60}")
                create_gif_from_frames(
                    str(full_input),
                    str(full_output),
                    fps=args.fps,
                    max_frames=args.max_frames,
                    skip_frames=args.skip
                )
            else:
                print(f"Skipping {input_dir} (directory not found)")
    
    elif args.input_dir:
        # Create GIF for specified directory
        input_path = Path(args.input_dir)
        if not input_path.is_absolute():
            input_path = project_dir / input_path
        
        if args.output:
            output_path = Path(args.output)
            if not output_path.is_absolute():
                output_path = project_dir / output_path
        else:
            # Generate output path based on input directory name
            dir_name = input_path.name
            output_path = project_dir / 'outputs' / f'{dir_name}.gif'
        
        create_gif_from_frames(
            str(input_path),
            str(output_path),
            fps=args.fps,
            max_frames=args.max_frames,
            skip_frames=args.skip
        )
    
    else:
        parser.print_help()
        print("\n" + "="*60)
        print("Examples:")
        print("  # Create GIFs for all frame directories:")
        print("  python create_gif.py --all")
        print()
        print("  # Create GIF from specific directory:")
        print("  python create_gif.py data/IMG_9182_frames")
        print()
        print("  # Create smaller GIF (every 2nd frame, 15 fps):")
        print("  python create_gif.py data/IMG_9182_frames --skip 2 --fps 15")
        print("="*60)


if __name__ == '__main__':
    main()
