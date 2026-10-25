"""Reassemble the published scan.npz byte-for-byte and verify SHA-256."""
from pathlib import Path
import hashlib
import json

def main():
    folder = Path(__file__).parent / 'results' / 'scan.npz.parts'
    manifest = json.loads((folder / 'manifest.json').read_text())
    chunks = []
    for part in manifest['parts']:
        chunk = (folder / part['name']).read_bytes()
        if len(chunk) != part['size'] or hashlib.sha256(chunk).hexdigest() != part['sha256']:
            raise ValueError('Invalid data part: ' + part['name'])
        chunks.append(chunk)
    data = b''.join(chunks)
    if len(data) != manifest['size'] or hashlib.sha256(data).hexdigest() != manifest['sha256']:
        raise ValueError('Reassembled data failed verification')
    target = folder.parent / manifest['output']
    target.write_bytes(data)
    print('Verified and restored:', target)

if __name__ == '__main__':
    main()
