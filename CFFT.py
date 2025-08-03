from Dataset import get_data
import numpy as np
import torch

def CFFT():
    aa_to_complex = {
        'A': complex(1.8, 0),  # Ala
        'C': complex(2.5, 0),  # Cys
        'D': complex(-3.5, -1),  # Asp
        'E': complex(-3.5, -1),  # Glu
        'F': complex(2.8, 0),  # Phe
        'G': complex(-0.4, 0),  # Gly
        'H': complex(-3.2, +1),  # His
        'I': complex(4.5, 0),  # Ile
        'K': complex(-3.9, +1),  # Lys
        'L': complex(3.8, 0),  # Leu
        'M': complex(1.9, 0),  # Met
        'N': complex(-3.5, 0),  # Asn
        'P': complex(-1.6, 0),  # Pro，
        'Q': complex(-3.5, 0),  # Gln
        'R': complex(-4.5, +1),  # Arg
        'S': complex(-0.8, 0),  # Ser
        'T': complex(-0.7, 0),  # Thr，
        'V': complex(4.2, 0),  # Val
        'W': complex(-0.9, 0),  # Trp
        'Y': complex(-1.3, 0),  # Tyr
    }

    def encode_complex(sequence):
        return [aa_to_complex[aa] for aa in sequence]

  
    numerical_vector = [encode_complex(seq) for seq in peptide_seq_list]

    
    max_length = max(len(seq) for seq in numerical_vector)
    numerical_vector_padding = [vector + [0] * (max_length - len(vector)) for vector in numerical_vector]

 
    fft_result = np.fft.fft(numerical_vector_padding)

  
    fft_magnitude = np.abs(fft_result)

 
    fft_phase = np.angle(fft_result)


    magnitude_to_phase_ratio = fft_magnitude / (np.abs(fft_phase) + 1e-6) 


    signal_feature = [fft_magnitude, fft_phase, magnitude_to_phase_ratio]

    signal_feature = np.stack(signal_feature, -1)

    signal_feature = torch.tensor(signal_feature)

    return signal_feature
