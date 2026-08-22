#!/usr/bin/env python3
"""Debug version parsing for armourycratecontrolinterface.inf"""
import re

# Read raw output
with open(r"D:\release-foundation\raw_drivers.txt", "r", encoding="utf-8", errors="ignore") as f:
    raw = f.read()

# Split into blocks
blocks = re.split(r'\r?\n\r?\n', raw)
blocks = [b for b in blocks if 'Published Name' in b or '发布名称' in b]

# Find armourycratecontrolinterface.inf entries
for b in blocks:
    if 'armourycratecontrolinterface.inf' in b.lower():
        print("=== Block ===")
        print(b)
        print()
        
        # Parse version
        m = re.search(r'Driver Version\s*:\s*(.+)', b)
        if m:
            version_str = m.group(1).strip()
            print(f"Raw version string: '{version_str}'")
            
            # Extract version number (after date)
            version_parts = version_str.split()
            if len(version_parts) >= 2:
                version_num = version_parts[-1]
                print(f"Version number: '{version_num}'")
                
                # Parse version
                version_clean = re.sub(r'[^\d.]', '', version_num)
                version_parsed = tuple(int(x) for x in version_clean.split('.') if x)
                print(f"Version parsed: {version_parsed}")
        print()
