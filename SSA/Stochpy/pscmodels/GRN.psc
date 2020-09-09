
# Keywords
Description: Gene-regulatory network
Modelname: GNR
Output_In_Conc: False
Species_In_Conc: False
 

# Reactions
R1:
    G1 > G1+M
    Kp*G1
R2:
    M > M+P
    Kt*M
R3:
    G1 + P > G0
    Kb*G1*P
R4:
    G0 > G1 + P
    Ku*G0
R5:
    M > $pool
    Kd1*M
R6:
    P > $pool
    Kd2*P
 
# Fixed species
 
# Variable species
G0 = 0.0
G1 = 1.0
M = 1.0
P = 100.0
 
# Parameters
Kp = 350.0
Kt = 0.001*30.0*10000.0
Kd1 = 0.001
Kd2 = 0.05*30
Kb = 166.0
Ku = 1.0

