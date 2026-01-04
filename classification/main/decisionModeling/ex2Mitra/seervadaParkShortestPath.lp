\ Model Seervada Park - Shortest Path
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  2 useLink_O_A + 5 useLink_O_B + 4 useLink_O_C + 7 useLink_A_D
   + 7 useLink_D_A + 2 useLink_A_B + 2 useLink_B_A + 4 useLink_B_D
   + 4 useLink_D_B + 3 useLink_B_E + 3 useLink_E_B + useLink_B_C
   + useLink_C_B + 4 useLink_C_E + 4 useLink_E_C + useLink_D_E
   + useLink_E_D + 5 useLink_D_T + 7 useLink_E_T
Subject To
 flowConservation_A: useLink_O_A - useLink_A_D + useLink_D_A - useLink_A_B
   + useLink_B_A = 0
 flowConservation_B: useLink_O_B + useLink_A_B - useLink_B_A - useLink_B_D
   + useLink_D_B - useLink_B_E + useLink_E_B - useLink_B_C + useLink_C_B
   = 0
 flowConservation_C: useLink_O_C + useLink_B_C - useLink_C_B - useLink_C_E
   + useLink_E_C = 0
 flowConservation_D: useLink_A_D - useLink_D_A + useLink_B_D - useLink_D_B
   - useLink_D_E + useLink_E_D - useLink_D_T = 0
 flowConservation_E: useLink_B_E - useLink_E_B + useLink_C_E - useLink_E_C
   + useLink_D_E - useLink_E_D - useLink_E_T = 0
 outgoing_0: useLink_O_A + useLink_O_B + useLink_O_C = 1
 ingoingT: - useLink_D_T - useLink_E_T = -1
Bounds
Binaries
 useLink_O_A useLink_O_B useLink_O_C useLink_A_D useLink_D_A useLink_A_B
 useLink_B_A useLink_B_D useLink_D_B useLink_B_E useLink_E_B useLink_B_C
 useLink_C_B useLink_C_E useLink_E_C useLink_D_E useLink_E_D useLink_D_T
 useLink_E_T
End
