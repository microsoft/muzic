# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

from tqdm import tqdm
import sys
if __name__ == '__main__':
    assert len(sys.argv) == 2
    name = sys.argv[1]
    prefix = f'data/{name}'
    for sp in ['train', 'valid', 'test']:
        with open(f'{prefix}/{sp}.trend', 'r') as src, open(f'{prefix}/{sp}.notes', 'r') as tgt,\
                open(f'{prefix}/{sp}.align', 'w') as align:
            for trend, notes in tqdm(list(zip(src, tgt))):
                l1 = len(trend.split())
                l2 = len(notes.split())
                assert (l1-1) // 3 == l2 // 4
                aligns = []
                for i in range(0, (l1-1) // 3):
                    # Tonality->Pitch
                    aligns.append(f'0-{4 * i + 2}')
                    # Chord->Pitch
                    aligns.append(f'{1 + 3 * i}-{4 * i + 2}')
                    # Cadence->Pitch
                    aligns.append(f'{1 + 3 * i + 1}-{4 * i + 2}')
                    # Cadence->Dur
                    aligns.append(f'{1 + 3 * i + 1}-{4 * i + 3}')
                    if 4 * i + 5 < l2:
                        # Cadence->Next Bar
                        aligns.append(f'{1 + 3 * i + 1}-{4 * (i + 1)}')
                        # Cadence->Next Pos
                        aligns.append(f'{1 + 3 * i + 1}-{4 * (i + 1) + 1}')
                    # Rhythm->Pos
                    aligns.append(f'{1 + 3 * i + 2}-{4 * i + 1}')
                print(" ".join(aligns), file=align)
