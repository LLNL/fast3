import sys
import numpy as np
import pandas as pd
import json
from Bio import SeqIO

from esm.models.esmc import ESMC # ESM Cambrian (gen model for protein representations)
from esm.models.esm3 import ESM3 # ESM3 (seq-seq BERD-based LLM)
from esm.sdk.api import ESM3InferenceClient, ESMProtein, SamplingConfig
from rdkit import Chem
from rdkit.Chem.Descriptors import CalcMolDescriptors
from pdb import set_trace
sys.stdout.flush()


#csv_fn = "conotoxin_seq.csv"
#out_fn = "conotoxin_esm3.npy"



# load pre-trained ESM model
model: ESM3InferenceClient = ESM3.from_pretrained("esm3-open").to("cuda")
#model = esm.sdk.client("esm3-small", token=my_token)

# iterate over sequence data

dataset = "dengue"
emb_dict = {}

global_names = []

def get_molecule_descriptors(smiles):
    global global_names
    # Convert SMILES string to RDKit molecule object
    molecule = Chem.MolFromSmiles(smiles)

    # Dictionary to store descriptor names and their values
    descriptors = CalcMolDescriptors(molecule)
    # List of descriptors to compute
    if not global_names:
        global_names = [desc_name for desc_name,_ in descriptors.items()]

    descriptor_list = [descriptors[desc_name] for desc_name in global_names]
    return np.array(descriptor_list)/np.linalg.norm(np.array(descriptor_list))

if dataset=="pdbbind":
    csv_fn = "data/pdbbind2020/scaled_descriptors/pdbbind_v2020_converted_train_valid_test_scaffold_with_rdkit_raw_descriptors.csv"
    out_fn = "embeddings/rdkit_raw_esm3_embeddings_casf_2016.json"
    csv_fn2 = "/p/vast1/bcwc/PDBBind/pdbbind_casf2016_str.csv"
    df = pd.read_csv(csv_fn)
    df2 = pd.read_csv(csv_fn2)
    pdbids = df2['pdbid'].to_list()
    df = df[df['pdbid'].isin(pdbids)]
    for ind, row in df.iterrows():
        pdbid = row["pdbid"]
        sequence = df2[df2['pdbid'] == row["pdbid"]]['amino_seq'].values[0]
        ligand_emb = row.to_numpy()[3:]
        ligand_emb = ligand_emb/np.linalg.norm(ligand_emb)
        protein = ESMProtein(sequence=(sequence))
        protein_tensor = model.encode(protein)
        protein_emb = model.forward_and_sample(protein_tensor, SamplingConfig(return_mean_embedding=True)).mean_embedding
        protein_emb = protein_emb.detach().cpu().numpy()
        protein_emb = protein_emb/np.linalg.norm(protein_emb)
        protein_ligand_embedding = np.concatenate([ligand_emb, protein_emb])
        print(sequence)
        print(protein_ligand_embedding.shape)
        emb_dict[pdbid] = (sequence, (protein_ligand_embedding).tolist())

if dataset =="dengue":
    dengue_protein = '/p/vast1/bcwc/flavivirus/data/flavivirus_docking_results/dengue/denv2/2fom/2fom_moe_prep.pdb'
    dengue_pdb = SeqIO.parse(dengue_protein, 'pdb-seqres')
    dengue_seq = min([str(pqseq.seq) for pqseq in dengue_pdb])    
    dengue_embed = ESMProtein(sequence=(dengue_seq))
    dengue_emb = model.encode(dengue_embed)
    dengue_emb = model.forward_and_sample(dengue_emb, SamplingConfig(return_mean_embedding=True)).mean_embedding
    dengue_emb = dengue_emb.detach().cpu().numpy()
    dengue_emb = dengue_emb/np.linalg.norm(dengue_emb)
    csv_fn = "data/dengue/denv2/scaled_descriptors/protease_ligand_prep_with_rdkit_raw_descriptors.csv"
    test_set = "data/dengue/denv2/train_test_valid_ids_random.csv"
    out_fn = "embeddings/protein-ligand-dengue-cut-normalized-rdkit-raw.json"
    df = pd.read_csv(csv_fn)
    df2 = pd.read_csv(test_set)
    df = df.drop(['base_rdkit_smiles'], axis=1)
    compound_ids = df2[df2["subset"] == "test"]["cmpd_id"].tolist()
    df = df[df['compound_id'].isin(compound_ids)]
    for ind, row in df.iterrows():
        did = row['compound_id']
        set_trace()
        ligand_emb = row.to_numpy()[2:]
        ligand_emb = ligand_emb/np.linalg.norm(ligand_emb)

        protein_ligand_embedding = np.concatenate([ligand_emb, dengue_emb])
        emb_dict[did] = (dengue_seq, (protein_ligand_embedding).tolist())
        print(dengue_seq)
        print(protein_ligand_embedding.shape)
with open(out_fn, 'w') as file:
    json.dump(emb_dict, file)