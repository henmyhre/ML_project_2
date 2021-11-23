from Bio import SeqIO

input_file = "NoGapsMSA_SIS1-on-top.fasta"
seq_true_file = "seq_true.fasta"
seq_false_file = "seq_false.fasta"
  
  
def split_sequence(sequence):
  seq_len = len(sequence)
  start, end = sequence[:int(seq_len/2)], sequence[int(seq_len/2):]
  return start, end
  
def split_file(input_file, true_file, false_file):
  
  fasta_sequences = SeqIO.parse(open(input_file),'fasta')
  
  seq_end_list = []
  seq_start_list = []
  
  with open(true_file, "w") as seq_true_file:
    for fasta in fasta_sequences:
      name, sequence = fasta.id, str(fasta.seq)
      sequence_start, sequence_end = split_sequence(sequence)
      seq_true_file.write(">" + name + "\n" + sequence_start + "\n" + sequence_end + "\n" + "1" + "\n")
      
      seq_start_list.append([name, sequence_start])
      seq_end_list.append([name, sequence_end])
    
    seq_true_file.close()
    
  with open(false_file, "w") as seq_false_file:    
    for start in seq_start_list:
      for end in seq_end_list:
        if start[0] != end[0]:
          seq_false_file.write(">" + start[0] + " | " + end[0] + "\n" + start[1] + "\n" + end[1] + "\n" +"0" +"\n")
        
    seq_false_file.close()
    
split_file(input_file, seq_true_file, seq_false_file)

