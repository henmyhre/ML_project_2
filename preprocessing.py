from Bio import SeqIO

input_file = "NoGapsMSA_SIS1-on-top.fasta"
seq_start_file = "seq_start.fasta"
seq_end_file = "seq_end.fasta"
  
  
def split_sequence(sequence):
  seq_len = len(sequence)
  start, end = sequence[:int(seq_len/2)], sequence[int(seq_len/2):]
  return start, end
  
def split_file(input_file, start_file, end_file):
  
  fasta_sequences = SeqIO.parse(open(input_file),'fasta')
  
  with open(start_file, "w") as seq_start_file, open(end_file, "w") as seq_end_file:
    for fasta in fasta_sequences:
        name, sequence = fasta.id, str(fasta.seq)
        sequence_start, sequence_end = split_sequence(sequence)
        seq_start_file.write(">" + name + "\n" + sequence_start + "\n")
        seq_end_file.write(">" + name + "\n" + sequence_end + "\n")
    seq_start_file.close()
    seq_end_file.close()
  
""" 
def shuffle_file(file_path):
  ### TODO
"""
    
split_file(input_file, seq_start_file, seq_end_file)
