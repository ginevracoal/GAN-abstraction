
# Keywords
Description: Loop of three genes that inibits each other (oscillating)
Modelname: Repressilator
Output_In_Conc: False
Species_In_Conc: False
 

# Reactions
R1:
    G1_on > G1_on + P1
    Kp1*G1_on
R2:
    G2_on > G2_on + P2
    Kp2*G2_on
R3:
    G3_on > G3_on + P3
    Kp3*G3_on
R4:
    P2 + G1_on > G1_off
    Kb1*G1_on*P2
R5:
    P3 + G2_on > G2_off
    Kb2*G2_on*P3
R6:
    P1 + G3_on > G3_off
    Kb3*G3_on*P1
R7:
    G1_off > P2 + G1_on
    Ku1*G1_off
R8:
    G2_off > P3 + G2_on
    Ku2*G2_off
R9:
    G3_off > P1 + G3_on
    Ku3*G3_off
R10:
    P1 > $pool
    Kd1*P1
R11:
    P2 > $pool
    Kd2*P2
R12:
    P3 > $pool
    Kd3*P3
 
# Fixed species
 
# Variable species
G1_on = 1.0
G1_off = 0.0
G2_on = 1.0
G2_off = 0.0
G3_on = 1.0
G3_off = 0.0
P1 = 0.0
P2 = 0.0
P3 = 0.0
 
# Parameters
Kp1 = 1.0
Kp2 = 1.0
Kp3 = 1.0
Kd1 = 0.01
Kd2 = 0.01
Kd3 = 0.01
Kb1 = 1.0
Kb2 = 1.0
Kb3 = 1.0
Ku1 = 1.0
Ku2 = 1.0
Ku3 = 1.0

