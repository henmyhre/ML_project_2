from Bio import SeqIO
from src.CONSTS import * 

def split_sequence(sequence):
    seq_len = len(sequence)
    start, end = sequence[:int(seq_len/2)], sequence[int(seq_len/2):]
    return start, end
  
def split_file(input_file, output_file):
  
    fasta_sequences = SeqIO.parse(open(input_file),'fasta')
    
    seq_start_list = []
    seq_end_list = []
    
    for fasta in fasta_sequences:
        name, sequence = fasta.id, str(fasta.seq)
        sequence_start, sequence_end = split_sequence(sequence)
        seq_start_list.append([name, sequence_start])
        seq_end_list.append([name, sequence_end])
        
    count_true = 0
    count_false = 0
    with open(output_file, "w") as seq_file:    
        for start in seq_start_list:
            for index, end in enumerate(seq_end_list):
                
                label = 0
                if start[0] == end[0]: 
                    label = 1
                    count_true += 1
                elif index % 1000 ==0 and start[0] != end[0]:   # Not making so much false data
                    count_false += 1
                    
                seq_file.write(start[0] + "<>" + end[0] + ";" + start[1] + ";" + end[1] + ";" + str(label) + "\n")
            
        seq_file.close()
        return count_true, count_false
