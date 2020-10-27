# Keywords
Description: X16
Modelname: X16
Output_In_Conc: False
Species_In_Conc: False
 

# Reactions
R1:
    G1 > G1 + P1
    a11*G1
R2:
    G1 + P1 > G1
    a12*G1*P1
R3:
    G2 > G2 + P1
    a21*G2
R4:
    P1 > $pool
    b1*P1
R5:
    G1 > G2
    gamma12*G1
R6:
    G2 > G1
    gamma21*G2

 
# Fixed species
G1 = 1.0
G2 = 1.0
P1 = 100.0 

# Variable species

# Parameters
a11 = 100.0
a12 = 2.0
a21 = 500.0
b1 = 2.0
gamma12 = 1.0
gamma21 = 1.0
 
