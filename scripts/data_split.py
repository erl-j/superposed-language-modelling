import glob
import random

SEED = 0

# set seed
random.seed(SEED)

data_path = '../midi_data/metamidi_dataset'
midi_files = glob.glob(data_path + '/**/*.mid', recursive=True)

# get md5s
md5s = [f.split('/')[-1].split('.')[0] for f in midi_files]

# take unique
md5s = list(set(md5s))

# shuffle
md5s = random.sample(md5s, len(md5s))

# split sizes
DEV_RATIO = 0.9
dev_md5s = md5s[:int(DEV_RATIO*len(md5s))]
tst_md5s = md5s[int(DEV_RATIO*len(md5s)):]

TRN_RATIO = 0.9
trn_md5s = dev_md5s[:int(TRN_RATIO*len(dev_md5s))]
val_md5s = dev_md5s[int(TRN_RATIO*len(dev_md5s)):]


# write to file
with open('./split/trn_md5s.txt', 'w') as f:
    f.writelines("\n".join(trn_md5s))
    
with open('./split/val_md5s.txt', 'w') as f:
    f.writelines("\n".join(val_md5s))

with open('./split/tst_md5s.txt', 'w') as f:
    f.writelines("\n".join(tst_md5s))



