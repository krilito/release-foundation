#!/usr/bin/env python3
"""Debug sorting for armourycratecontrolinterface.inf"""
import re
from collections import defaultdict

# Read raw output
with open(r"D:\release-foundation\raw_drivers.txt", "r", encoding="utf-8", errors="ignore") as f:
    raw = f.read()

# Split into blocks
blocks = re.split(r'\r?\n\r?\n', raw)
blocks = [b for b in blocks if 'Published Name' in b or '发布名称' in b]

entries = []
for b in blocks:
    h = {}
    for line in b.split('\n'):
        m = re.match(r'^\s*([^:]+?)\s*:\s*(.+?)\s*$', line)
        if m:
            h[m.group(1).strip()] = m.group(2).strip()
    
    published = h.get('Published Name') or h.get('发布名称', '')
    original = h.get('Original Name') or h.get('原始名称', '')
    version = h.get('Driver Version') or h.get('驱动程序版本', '')
    
    if not published:
        continue
    
    # Parse version
    version_clean = re.sub(r'[^\d.]', '', version)
    version_parts = []
    for part in version_clean.split('.'):
        if part:
            version_parts.append(int(part))
    version_parsed = tuple(version_parts) if version_parts else (0,)
    
    entries.append({
        'PublishedName': published,
        'OriginalName': original.lower() if original else '',
        'Version': version,
        'VersionParsed': version_parsed
    })

# Find armourycratecontrolinterface.inf entries
inf_entries = [e for e in entries if e['OriginalName'] == 'armourycratecontrolinterface.inf']

print("=== Before sorting ===")
for e in inf_entries:
    print(f"  {e['PublishedName']}: {e['Version']} -> {e['VersionParsed']}")

# Sort
inf_entries.sort(key=lambda x: x['VersionParsed'], reverse=True)

print("\n=== After sorting (descending) ===")
for e in inf_entries:
    print(f"  {e['PublishedName']}: {e['Version']} -> {e['VersionParsed']}")

print(f"\nKeep: {inf_entries[0]['PublishedName']} ({inf_entries[0]['Version']})")
print(f"Delete: {inf_entries[1]['PublishedName']} ({inf_entries[1]['Version']})")
