from Bio import SeqIO

input_file = "NoGapsMSA_SIS1-on-top.fasta"
seq_true_file = "seq_true.csv"
seq_false_file = "seq_false.csv"
  
  
def split_sequence(sequence):
  seq_len = len(sequence)
  start, end = sequence[:int(seq_len/2)], sequence[int(seq_len/2):]
  return start, end
  
def split_file(input_file, true_file, false_file):
  
  fasta_sequences = SeqIO.parse(open(input_file),'fasta')
  
  seq_start_list = []
  seq_end_list = []
  
  with open(true_file, "w") as seq_true_file:
    for fasta in fasta_sequences:
      name, sequence = fasta.id, str(fasta.seq)
      sequence_start, sequence_end = split_sequence(sequence)
      seq_true_file.write(name + ";" + sequence_start + ";" + sequence_end + ";1\n")
      
      seq_start_list.append([name, sequence_start])
      seq_end_list.append([name, sequence_end])
    
    seq_true_file.close()
    
  with open(false_file, "w") as seq_false_file:    
    for start in seq_start_list:
      for end in seq_end_list:
        if start[0] == end[0]: continue
        seq_false_file.write(start[0] + "<>" + end[0] + ";" + start[1] + ";" + end[1] + ";0\n")
        
    seq_false_file.close()
    
split_file(input_file, seq_true_file, seq_false_file)

