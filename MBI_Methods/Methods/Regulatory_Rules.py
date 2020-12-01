import numpy as np 
import pdb

def Give_Regulatory_Network(T_Window, avg_Reaction_Propensity):
        A_birth, B_Birth, A_death, B_death, B_Down_A, A_Down_B, B_Up_A, A_Up, B_Up, A_Up_B = avg_Reaction_Propensity
        
        Threshold = 10.0/T_Window # atleast 10 events per second
        
        Reg_Net = np.zeros((2,2))
        
        if A_Up > Threshold:
                Reg_Net[0,0] =  A_Up/A_death # A_Up - A_death
        
        print("No a Up")
        
        if B_Up > Threshold:
                Reg_Net[1,1] =  B_Up/B_death # B_Up - B_death
        
        if  np.abs(-B_Down_A + B_Up_A) > Threshold:
                Reg_Net[0,1] = -B_Down_A + B_Up_A

        if np.abs(-A_Down_B + A_Up_B) > Threshold:
                Reg_Net[1,0] = -A_Down_B + A_Up_B

        return Reg_Net