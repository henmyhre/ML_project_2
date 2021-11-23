from Bio import SeqIO

input_file = "NoGapsMSA_SIS1-on-top.fasta"
seq_start_file = "seq_start.fasta"
seq_end_file = "seq_end.fasta"
shuffled_seg_end_file = "shuffled_seq_end.fasta"
  
  
def split_sequence(sequence):
  seq_len = len(sequence)
  start, end = sequence[:int(seq_len/2)], sequence[int(seq_len/2):]
  return start, end
  
def split_file(input_file, start_file, end_file, shuffled_end_file):
  
  fasta_sequences = SeqIO.parse(open(input_file),'fasta')
  
  seq_end_list = []
  
  with open(start_file, "w") as seq_start_file:
    for fasta in fasta_sequences:
      name, sequence = fasta.id, str(fasta.seq)
      sequence_start, sequence_end = split_sequence(sequence)
      seq_start_file.write(">" + name + "\n" + sequence_start + "\n")
      
      seq_end_list.append([name, sequence_end])
    
    seq_start_file.close()
    
  with open(end_file, "w") as seq_end_file, open(shuffled_end_file, "w") as shuffled_seq_end_file:
    len_seq_end_list = len(seq_end_list)
    
    for i in range(len_seq_end_list):
      seq_end_file.write(">" + seq_end_list[i][0] + "\n" + seq_end_list[i][1] + "\n")
      
    for i in range(1, len_seq_end_list):
      for j in range(len_seq_end_list):
        shuffled_seq_end_file.write(">" + seq_end_list[(i+j)%len_seq_end_list][0] + "\n" + seq_end_list[(i+j)%len_seq_end_list][1] + "\n")
        
    seq_end_file.close()
    shuffled_seq_end_file.close()
    
split_file(input_file, seq_start_file, seq_end_file, shuffled_seg_end_file)
