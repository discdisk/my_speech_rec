import os 

for u in [512,1024]:
    for n in [5,7]:
        epoch = 5
        unit_size = u
        n_layers = n
        b_size = 20 if unit_size>600 else 30

        os.system(f'python CTCwithAttention.py --e {epoch} --b {b_size} --u {unit_size} --n {n_layers}')