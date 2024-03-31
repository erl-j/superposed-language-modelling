#%%
import glob
import random
import pandas

SEED = 0

# set seed
random.seed(SEED)

DEV_RATIO = 0.9
TRN_RATIO = 0.9

# MMD_audio_text_matches.tsv
matches_df = pandas.read_csv('../midi_data/metamidi_dataset/MMD_audio_matches.tsv', sep='\t')

# count unique md5s

print(len(matches_df["md5"].unique()))

# add md5 to all md5s
matches_df["md5_"] = matches_df["md5"].apply(lambda x: f"md5_:{x}")

# add sid to all sids
matches_df["sid_"] = matches_df["sid"].apply(lambda x: f"sid_:{x}")


import networkx as nx

G = nx.Graph()
G.add_edges_from(matches_df[["md5_", "sid_"]].to_numpy().tolist())

print(len(list(nx.connected_components(G))))

# list of cc in random order
ccs = list(nx.connected_components(G))
ccs = random.sample(ccs, len(ccs))

# split into dev, val and test
dev_ccs = ccs[:int(DEV_RATIO*len(ccs))]
tst_ccs = ccs[int(DEV_RATIO*len(ccs)):]

# split dev into trn and val
trn_ccs = dev_ccs[:int(TRN_RATIO*len(dev_ccs))]
val_ccs = dev_ccs[int(TRN_RATIO*len(dev_ccs)):]

trn_cc_md5s = []
for cc in trn_ccs:
    for node in cc:
        if node.startswith("md5_:"):
            trn_cc_md5s.append(node.replace("md5_:", ""))

val_cc_md5s = []
for cc in val_ccs:
    for node in cc:
        if node.startswith("md5_:"):
            val_cc_md5s.append(node.replace("md5_:", ""))

tst_cc_md5s = []
for cc in tst_ccs:
    for node in cc:
        if node.startswith("md5_:"):
            tst_cc_md5s.append(node.replace("md5_:", ""))


#%%

# set seed
random.seed(SEED)

data_path = '../midi_data/metamidi_dataset'
midi_files = glob.glob(data_path + '/**/*.mid', recursive=True)

# get md5s
md5s = [f.split('/')[-1].split('.')[0] for f in midi_files]


match_md5s = matches_df["md5"].unique()
# turn into set
match_md5s = set(match_md5s)

# get md5s not in matches
md5s = [md5 for md5 in md5s if md5 not in match_md5s]

# take unique
md5s = list(set(md5s))

# shuffle
md5s = random.sample(md5s, len(md5s))

# split sizes
dev_md5s = md5s[:int(DEV_RATIO*len(md5s))]
tst_md5s = md5s[int(DEV_RATIO*len(md5s)):]


trn_md5s = dev_md5s[:int(TRN_RATIO*len(dev_md5s))]
val_md5s = dev_md5s[int(TRN_RATIO*len(dev_md5s)):]


# add cc md5s
trn_md5s.extend(trn_cc_md5s)
val_md5s.extend(val_cc_md5s)
tst_md5s.extend(tst_cc_md5s)

# verify no overlap
assert len(set(trn_md5s).intersection(set(val_md5s))) == 0
assert len(set(trn_md5s).intersection(set(tst_md5s))) == 0
assert len(set(val_md5s).intersection(set(tst_md5s))) == 0

# print sizes
print(f"trn: {len(trn_md5s)}")
print(f"val: {len(val_md5s)}")
print(f"tst: {len(tst_md5s)}")

# write to file
with open('./split/trn_md5s.txt', 'w') as f:
    f.writelines("\n".join(trn_md5s))
    
with open('./split/val_md5s.txt', 'w') as f:
    f.writelines("\n".join(val_md5s))

with open('./split/tst_md5s.txt', 'w') as f:
    f.writelines("\n".join(tst_md5s))



