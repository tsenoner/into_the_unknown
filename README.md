## run foldseek all-vs-all swissprot

### high sensitivity
foldseek easy-search afdb_swissprot_raw_fcz.tar afdb_swissprot_raw_fcz.tar result.m8 tmp --input-format 5 --threads 4 -s 9.0 --exhaustive-search --max-seqs 10000 --cov-mode 0 --format-mode 4 --format-output query,target,fident,nident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,cigar,qcov,tcov,lddt,rmsd,alntmscore,qtmscore,ttmscore

### low sensitivity - fast
foldseek easy-search afdb_swissprot_raw_fcz.tar afdb_swissprot_raw_fcz.tar result.m8 tmp --input-format 5 --threads 4 -s 1.0 --cov-mode 0 --format-mode 4 --format-output query,target,fident,nident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,cigar,qcov,tcov,lddt,rmsd,alntmscore,qtmscore,ttmscore

## Foldcomp
- extract pLDDT: `bin/foldcomp extract --plddt -t 4 data/swissprot/foldcomp/afdb_swissprot_v4/afdb_swissprot_v4`

## get sequences entered in TrEMBL after 2024
- sequences entered after 2024: `(reviewed:false) AND (date_created:[2024-01-01 TO 2024-06-04])` -> 9,933,238 proteins
- UniRef50 sequences last updated after 2024: `(created:[2024-01-01 TO 2024-06-04]) AND (identity:0.5)` -> 20,845,580 clusters
- run `into_the_unknown/prepare_data/parse_uniprot_after2024.py`
- run `into_the_unknown/prepare_data/get_new_uniref50_clusters.py` -> 2,276,776 clusters
- remove UniParc entries -> 1,611,802 clusters
- remove sequences with **non-terminal** residues and **fragments** -> 1,448,114 clusters - 1,780,288 proteins
- remove **duplicate** sequences -> 1,773,081 proteins
- dataset 1 (filter by cluster size and organism size):
  - remove clusters that have **less then 3 members** -> 57,558 clusters - 271,234 proteins
  - remove clusters that have **less then 2 different species** types -> 12,120 clusters - *52,303 proteins*
- dataset 2 (Protein exists):
  - Filter for **Protein Existance** sequence:
    - "Evidence at protein level": 1
    - "Evidence at transcript level": 2
    - *326 proteins*

## Files
- `into_the_unknown/prepare_data/parse_uniprot_after2024.py`:
  - parses JSON file from UniProt query: `(date_created:[2024-01-01 TO 2024-06-04]) AND (reviewed:false)`
  - extracts columns:
    - uid (str)
    - taxon_id (int)
    - protein_existance (int)
    - fragment (bool)
    - non_terminal (bool)
    - seq (str)
- `into_the_unknown/prepare_data/get_new_uniref50_clusters.py`:
  - Create a set of UniProt IDs entered after 2024
  - Filter for UniRef50 clusters
    - containing only proteins entered in UniProt after 2024
    - With the UniParc ID larger to **UPI002B50703F** (First entry in 2024)