import pandas as pd
df = pd.read_csv("cpgIslandExt.txt", sep="\t", header=None, usecols=[1,2,3])
df.to_csv("cpg_sites.bed", sep="\t", header=False, index=False)
