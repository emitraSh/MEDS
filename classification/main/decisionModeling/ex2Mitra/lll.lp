\ Model Minimum Spanning Tree of Tree
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  1.3 link_1,2 + 2.1 link_1,3 + 0.9 link_1,4 + 0.7 link_1,5 + 1.8 link_1,6
   + 2 link_1,7 + 1.5 link_1,8 + 0.9 link_2,3 + 1.8 link_2,4 + 1.2 link_2,5
   + 2.6 link_2,6 + 2.3 link_2,7 + 1.1 link_2,8 + 2.6 link_3,4
   + 1.7 link_3,5 + 2.5 link_3,6 + 1.9 link_3,7 + link_3,8 + 0.7 link_4,5
   + 1.6 link_4,6 + 1.5 link_4,7 + 0.9 link_4,8 + 0.9 link_5,6
   + 1.1 link_5,7 + 0.8 link_5,8 + 0.6 link_6,7 + link_6,8 + 0.5 link_7,8
Subject To
 maxNodesConstr: link_1,2 + link_1,3 + link_1,4 + link_1,5 + link_1,6
   + link_1,7 + link_1,8 + link_2,3 + link_2,4 + link_2,5 + link_2,6
   + link_2,7 + link_2,8 + link_3,4 + link_3,5 + link_3,6 + link_3,7
   + link_3,8 + link_4,5 + link_4,6 + link_4,7 + link_4,8 + link_5,6
   + link_5,7 + link_5,8 + link_6,7 + link_6,8 + link_7,8 = 7
 s: link_1,2 <= 1
 s: link_1,3 <= 1
 s: link_1,4 <= 1
 s: link_1,5 <= 1
 s: link_1,6 <= 1
 s: link_1,7 <= 1
 s: link_1,8 <= 1
 s: link_2,3 <= 1
 s: link_2,4 <= 1
 s: link_2,5 <= 1
 s: link_2,6 <= 1
 s: link_2,7 <= 1
 s: link_2,8 <= 1
 s: link_3,4 <= 1
 s: link_3,5 <= 1
 s: link_3,6 <= 1
 s: link_3,7 <= 1
 s: link_3,8 <= 1
 s: link_4,5 <= 1
 s: link_4,6 <= 1
 s: link_4,7 <= 1
 s: link_4,8 <= 1
 s: link_5,6 <= 1
 s: link_5,7 <= 1
 s: link_5,8 <= 1
 s: link_6,7 <= 1
 s: link_6,8 <= 1
 s: link_7,8 <= 1
 s: link_1,2 + link_1,3 + link_2,3 <= 2
 s: link_1,2 + link_1,4 + link_2,4 <= 2
 s: link_1,2 + link_1,5 + link_2,5 <= 2
 s: link_1,2 + link_1,6 + link_2,6 <= 2
 s: link_1,2 + link_1,7 + link_2,7 <= 2
 s: link_1,2 + link_1,8 + link_2,8 <= 2
 s: link_1,3 + link_1,4 + link_3,4 <= 2
 s: link_1,3 + link_1,5 + link_3,5 <= 2
 s: link_1,3 + link_1,6 + link_3,6 <= 2
 s: link_1,3 + link_1,7 + link_3,7 <= 2
 s: link_1,3 + link_1,8 + link_3,8 <= 2
 s: link_1,4 + link_1,5 + link_4,5 <= 2
 s: link_1,4 + link_1,6 + link_4,6 <= 2
 s: link_1,4 + link_1,7 + link_4,7 <= 2
 s: link_1,4 + link_1,8 + link_4,8 <= 2
 s: link_1,5 + link_1,6 + link_5,6 <= 2
 s: link_1,5 + link_1,7 + link_5,7 <= 2
 s: link_1,5 + link_1,8 + link_5,8 <= 2
 s: link_1,6 + link_1,7 + link_6,7 <= 2
 s: link_1,6 + link_1,8 + link_6,8 <= 2
 s: link_1,7 + link_1,8 + link_7,8 <= 2
 s: link_2,3 + link_2,4 + link_3,4 <= 2
 s: link_2,3 + link_2,5 + link_3,5 <= 2
 s: link_2,3 + link_2,6 + link_3,6 <= 2
 s: link_2,3 + link_2,7 + link_3,7 <= 2
 s: link_2,3 + link_2,8 + link_3,8 <= 2
 s: link_2,4 + link_2,5 + link_4,5 <= 2
 s: link_2,4 + link_2,6 + link_4,6 <= 2
 s: link_2,4 + link_2,7 + link_4,7 <= 2
 s: link_2,4 + link_2,8 + link_4,8 <= 2
 s: link_2,5 + link_2,6 + link_5,6 <= 2
 s: link_2,5 + link_2,7 + link_5,7 <= 2
 s: link_2,5 + link_2,8 + link_5,8 <= 2
 s: link_2,6 + link_2,7 + link_6,7 <= 2
 s: link_2,6 + link_2,8 + link_6,8 <= 2
 s: link_2,7 + link_2,8 + link_7,8 <= 2
 s: link_3,4 + link_3,5 + link_4,5 <= 2
 s: link_3,4 + link_3,6 + link_4,6 <= 2
 s: link_3,4 + link_3,7 + link_4,7 <= 2
 s: link_3,4 + link_3,8 + link_4,8 <= 2
 s: link_3,5 + link_3,6 + link_5,6 <= 2
 s: link_3,5 + link_3,7 + link_5,7 <= 2
 s: link_3,5 + link_3,8 + link_5,8 <= 2
 s: link_3,6 + link_3,7 + link_6,7 <= 2
 s: link_3,6 + link_3,8 + link_6,8 <= 2
 s: link_3,7 + link_3,8 + link_7,8 <= 2
 s: link_4,5 + link_4,6 + link_5,6 <= 2
 s: link_4,5 + link_4,7 + link_5,7 <= 2
 s: link_4,5 + link_4,8 + link_5,8 <= 2
 s: link_4,6 + link_4,7 + link_6,7 <= 2
 s: link_4,6 + link_4,8 + link_6,8 <= 2
 s: link_4,7 + link_4,8 + link_7,8 <= 2
 s: link_5,6 + link_5,7 + link_6,7 <= 2
 s: link_5,6 + link_5,8 + link_6,8 <= 2
 s: link_5,7 + link_5,8 + link_7,8 <= 2
 s: link_6,7 + link_6,8 + link_7,8 <= 2
 s: link_1,2 + link_1,3 + link_1,4 + link_2,3 + link_2,4 + link_3,4 <= 3
 s: link_1,2 + link_1,3 + link_1,5 + link_2,3 + link_2,5 + link_3,5 <= 3
 s: link_1,2 + link_1,3 + link_1,6 + link_2,3 + link_2,6 + link_3,6 <= 3
 s: link_1,2 + link_1,3 + link_1,7 + link_2,3 + link_2,7 + link_3,7 <= 3
 s: link_1,2 + link_1,3 + link_1,8 + link_2,3 + link_2,8 + link_3,8 <= 3
 s: link_1,2 + link_1,4 + link_1,5 + link_2,4 + link_2,5 + link_4,5 <= 3
 s: link_1,2 + link_1,4 + link_1,6 + link_2,4 + link_2,6 + link_4,6 <= 3
 s: link_1,2 + link_1,4 + link_1,7 + link_2,4 + link_2,7 + link_4,7 <= 3
 s: link_1,2 + link_1,4 + link_1,8 + link_2,4 + link_2,8 + link_4,8 <= 3
 s: link_1,2 + link_1,5 + link_1,6 + link_2,5 + link_2,6 + link_5,6 <= 3
 s: link_1,2 + link_1,5 + link_1,7 + link_2,5 + link_2,7 + link_5,7 <= 3
 s: link_1,2 + link_1,5 + link_1,8 + link_2,5 + link_2,8 + link_5,8 <= 3
 s: link_1,2 + link_1,6 + link_1,7 + link_2,6 + link_2,7 + link_6,7 <= 3
 s: link_1,2 + link_1,6 + link_1,8 + link_2,6 + link_2,8 + link_6,8 <= 3
 s: link_1,2 + link_1,7 + link_1,8 + link_2,7 + link_2,8 + link_7,8 <= 3
 s: link_1,3 + link_1,4 + link_1,5 + link_3,4 + link_3,5 + link_4,5 <= 3
 s: link_1,3 + link_1,4 + link_1,6 + link_3,4 + link_3,6 + link_4,6 <= 3
 s: link_1,3 + link_1,4 + link_1,7 + link_3,4 + link_3,7 + link_4,7 <= 3
 s: link_1,3 + link_1,4 + link_1,8 + link_3,4 + link_3,8 + link_4,8 <= 3
 s: link_1,3 + link_1,5 + link_1,6 + link_3,5 + link_3,6 + link_5,6 <= 3
 s: link_1,3 + link_1,5 + link_1,7 + link_3,5 + link_3,7 + link_5,7 <= 3
 s: link_1,3 + link_1,5 + link_1,8 + link_3,5 + link_3,8 + link_5,8 <= 3
 s: link_1,3 + link_1,6 + link_1,7 + link_3,6 + link_3,7 + link_6,7 <= 3
 s: link_1,3 + link_1,6 + link_1,8 + link_3,6 + link_3,8 + link_6,8 <= 3
 s: link_1,3 + link_1,7 + link_1,8 + link_3,7 + link_3,8 + link_7,8 <= 3
 s: link_1,4 + link_1,5 + link_1,6 + link_4,5 + link_4,6 + link_5,6 <= 3
 s: link_1,4 + link_1,5 + link_1,7 + link_4,5 + link_4,7 + link_5,7 <= 3
 s: link_1,4 + link_1,5 + link_1,8 + link_4,5 + link_4,8 + link_5,8 <= 3
 s: link_1,4 + link_1,6 + link_1,7 + link_4,6 + link_4,7 + link_6,7 <= 3
 s: link_1,4 + link_1,6 + link_1,8 + link_4,6 + link_4,8 + link_6,8 <= 3
 s: link_1,4 + link_1,7 + link_1,8 + link_4,7 + link_4,8 + link_7,8 <= 3
 s: link_1,5 + link_1,6 + link_1,7 + link_5,6 + link_5,7 + link_6,7 <= 3
 s: link_1,5 + link_1,6 + link_1,8 + link_5,6 + link_5,8 + link_6,8 <= 3
 s: link_1,5 + link_1,7 + link_1,8 + link_5,7 + link_5,8 + link_7,8 <= 3
 s: link_1,6 + link_1,7 + link_1,8 + link_6,7 + link_6,8 + link_7,8 <= 3
 s: link_2,3 + link_2,4 + link_2,5 + link_3,4 + link_3,5 + link_4,5 <= 3
 s: link_2,3 + link_2,4 + link_2,6 + link_3,4 + link_3,6 + link_4,6 <= 3
 s: link_2,3 + link_2,4 + link_2,7 + link_3,4 + link_3,7 + link_4,7 <= 3
 s: link_2,3 + link_2,4 + link_2,8 + link_3,4 + link_3,8 + link_4,8 <= 3
 s: link_2,3 + link_2,5 + link_2,6 + link_3,5 + link_3,6 + link_5,6 <= 3
 s: link_2,3 + link_2,5 + link_2,7 + link_3,5 + link_3,7 + link_5,7 <= 3
 s: link_2,3 + link_2,5 + link_2,8 + link_3,5 + link_3,8 + link_5,8 <= 3
 s: link_2,3 + link_2,6 + link_2,7 + link_3,6 + link_3,7 + link_6,7 <= 3
 s: link_2,3 + link_2,6 + link_2,8 + link_3,6 + link_3,8 + link_6,8 <= 3
 s: link_2,3 + link_2,7 + link_2,8 + link_3,7 + link_3,8 + link_7,8 <= 3
 s: link_2,4 + link_2,5 + link_2,6 + link_4,5 + link_4,6 + link_5,6 <= 3
 s: link_2,4 + link_2,5 + link_2,7 + link_4,5 + link_4,7 + link_5,7 <= 3
 s: link_2,4 + link_2,5 + link_2,8 + link_4,5 + link_4,8 + link_5,8 <= 3
 s: link_2,4 + link_2,6 + link_2,7 + link_4,6 + link_4,7 + link_6,7 <= 3
 s: link_2,4 + link_2,6 + link_2,8 + link_4,6 + link_4,8 + link_6,8 <= 3
 s: link_2,4 + link_2,7 + link_2,8 + link_4,7 + link_4,8 + link_7,8 <= 3
 s: link_2,5 + link_2,6 + link_2,7 + link_5,6 + link_5,7 + link_6,7 <= 3
 s: link_2,5 + link_2,6 + link_2,8 + link_5,6 + link_5,8 + link_6,8 <= 3
 s: link_2,5 + link_2,7 + link_2,8 + link_5,7 + link_5,8 + link_7,8 <= 3
 s: link_2,6 + link_2,7 + link_2,8 + link_6,7 + link_6,8 + link_7,8 <= 3
 s: link_3,4 + link_3,5 + link_3,6 + link_4,5 + link_4,6 + link_5,6 <= 3
 s: link_3,4 + link_3,5 + link_3,7 + link_4,5 + link_4,7 + link_5,7 <= 3
 s: link_3,4 + link_3,5 + link_3,8 + link_4,5 + link_4,8 + link_5,8 <= 3
 s: link_3,4 + link_3,6 + link_3,7 + link_4,6 + link_4,7 + link_6,7 <= 3
 s: link_3,4 + link_3,6 + link_3,8 + link_4,6 + link_4,8 + link_6,8 <= 3
 s: link_3,4 + link_3,7 + link_3,8 + link_4,7 + link_4,8 + link_7,8 <= 3
 s: link_3,5 + link_3,6 + link_3,7 + link_5,6 + link_5,7 + link_6,7 <= 3
 s: link_3,5 + link_3,6 + link_3,8 + link_5,6 + link_5,8 + link_6,8 <= 3
 s: link_3,5 + link_3,7 + link_3,8 + link_5,7 + link_5,8 + link_7,8 <= 3
 s: link_3,6 + link_3,7 + link_3,8 + link_6,7 + link_6,8 + link_7,8 <= 3
 s: link_4,5 + link_4,6 + link_4,7 + link_5,6 + link_5,7 + link_6,7 <= 3
 s: link_4,5 + link_4,6 + link_4,8 + link_5,6 + link_5,8 + link_6,8 <= 3
 s: link_4,5 + link_4,7 + link_4,8 + link_5,7 + link_5,8 + link_7,8 <= 3
 s: link_4,6 + link_4,7 + link_4,8 + link_6,7 + link_6,8 + link_7,8 <= 3
 s: link_5,6 + link_5,7 + link_5,8 + link_6,7 + link_6,8 + link_7,8 <= 3
 s: link_1,2 + link_1,3 + link_1,4 + link_1,5 + link_2,3 + link_2,4
   + link_2,5 + link_3,4 + link_3,5 + link_4,5 <= 4
 s: link_1,2 + link_1,3 + link_1,4 + link_1,6 + link_2,3 + link_2,4
   + link_2,6 + link_3,4 + link_3,6 + link_4,6 <= 4
 s: link_1,2 + link_1,3 + link_1,4 + link_1,7 + link_2,3 + link_2,4
   + link_2,7 + link_3,4 + link_3,7 + link_4,7 <= 4
 s: link_1,2 + link_1,3 + link_1,4 + link_1,8 + link_2,3 + link_2,4
   + link_2,8 + link_3,4 + link_3,8 + link_4,8 <= 4
 s: link_1,2 + link_1,3 + link_1,5 + link_1,6 + link_2,3 + link_2,5
   + link_2,6 + link_3,5 + link_3,6 + link_5,6 <= 4
 s: link_1,2 + link_1,3 + link_1,5 + link_1,7 + link_2,3 + link_2,5
   + link_2,7 + link_3,5 + link_3,7 + link_5,7 <= 4
 s: link_1,2 + link_1,3 + link_1,5 + link_1,8 + link_2,3 + link_2,5
   + link_2,8 + link_3,5 + link_3,8 + link_5,8 <= 4
 s: link_1,2 + link_1,3 + link_1,6 + link_1,7 + link_2,3 + link_2,6
   + link_2,7 + link_3,6 + link_3,7 + link_6,7 <= 4
 s: link_1,2 + link_1,3 + link_1,6 + link_1,8 + link_2,3 + link_2,6
   + link_2,8 + link_3,6 + link_3,8 + link_6,8 <= 4
 s: link_1,2 + link_1,3 + link_1,7 + link_1,8 + link_2,3 + link_2,7
   + link_2,8 + link_3,7 + link_3,8 + link_7,8 <= 4
 s: link_1,2 + link_1,4 + link_1,5 + link_1,6 + link_2,4 + link_2,5
   + link_2,6 + link_4,5 + link_4,6 + link_5,6 <= 4
 s: link_1,2 + link_1,4 + link_1,5 + link_1,7 + link_2,4 + link_2,5
   + link_2,7 + link_4,5 + link_4,7 + link_5,7 <= 4
 s: link_1,2 + link_1,4 + link_1,5 + link_1,8 + link_2,4 + link_2,5
   + link_2,8 + link_4,5 + link_4,8 + link_5,8 <= 4
 s: link_1,2 + link_1,4 + link_1,6 + link_1,7 + link_2,4 + link_2,6
   + link_2,7 + link_4,6 + link_4,7 + link_6,7 <= 4
 s: link_1,2 + link_1,4 + link_1,6 + link_1,8 + link_2,4 + link_2,6
   + link_2,8 + link_4,6 + link_4,8 + link_6,8 <= 4
 s: link_1,2 + link_1,4 + link_1,7 + link_1,8 + link_2,4 + link_2,7
   + link_2,8 + link_4,7 + link_4,8 + link_7,8 <= 4
 s: link_1,2 + link_1,5 + link_1,6 + link_1,7 + link_2,5 + link_2,6
   + link_2,7 + link_5,6 + link_5,7 + link_6,7 <= 4
 s: link_1,2 + link_1,5 + link_1,6 + link_1,8 + link_2,5 + link_2,6
   + link_2,8 + link_5,6 + link_5,8 + link_6,8 <= 4
 s: link_1,2 + link_1,5 + link_1,7 + link_1,8 + link_2,5 + link_2,7
   + link_2,8 + link_5,7 + link_5,8 + link_7,8 <= 4
 s: link_1,2 + link_1,6 + link_1,7 + link_1,8 + link_2,6 + link_2,7
   + link_2,8 + link_6,7 + link_6,8 + link_7,8 <= 4
 s: link_1,3 + link_1,4 + link_1,5 + link_1,6 + link_3,4 + link_3,5
   + link_3,6 + link_4,5 + link_4,6 + link_5,6 <= 4
 s: link_1,3 + link_1,4 + link_1,5 + link_1,7 + link_3,4 + link_3,5
   + link_3,7 + link_4,5 + link_4,7 + link_5,7 <= 4
 s: link_1,3 + link_1,4 + link_1,5 + link_1,8 + link_3,4 + link_3,5
   + link_3,8 + link_4,5 + link_4,8 + link_5,8 <= 4
 s: link_1,3 + link_1,4 + link_1,6 + link_1,7 + link_3,4 + link_3,6
   + link_3,7 + link_4,6 + link_4,7 + link_6,7 <= 4
 s: link_1,3 + link_1,4 + link_1,6 + link_1,8 + link_3,4 + link_3,6
   + link_3,8 + link_4,6 + link_4,8 + link_6,8 <= 4
 s: link_1,3 + link_1,4 + link_1,7 + link_1,8 + link_3,4 + link_3,7
   + link_3,8 + link_4,7 + link_4,8 + link_7,8 <= 4
 s: link_1,3 + link_1,5 + link_1,6 + link_1,7 + link_3,5 + link_3,6
   + link_3,7 + link_5,6 + link_5,7 + link_6,7 <= 4
 s: link_1,3 + link_1,5 + link_1,6 + link_1,8 + link_3,5 + link_3,6
   + link_3,8 + link_5,6 + link_5,8 + link_6,8 <= 4
 s: link_1,3 + link_1,5 + link_1,7 + link_1,8 + link_3,5 + link_3,7
   + link_3,8 + link_5,7 + link_5,8 + link_7,8 <= 4
 s: link_1,3 + link_1,6 + link_1,7 + link_1,8 + link_3,6 + link_3,7
   + link_3,8 + link_6,7 + link_6,8 + link_7,8 <= 4
 s: link_1,4 + link_1,5 + link_1,6 + link_1,7 + link_4,5 + link_4,6
   + link_4,7 + link_5,6 + link_5,7 + link_6,7 <= 4
 s: link_1,4 + link_1,5 + link_1,6 + link_1,8 + link_4,5 + link_4,6
   + link_4,8 + link_5,6 + link_5,8 + link_6,8 <= 4
 s: link_1,4 + link_1,5 + link_1,7 + link_1,8 + link_4,5 + link_4,7
   + link_4,8 + link_5,7 + link_5,8 + link_7,8 <= 4
 s: link_1,4 + link_1,6 + link_1,7 + link_1,8 + link_4,6 + link_4,7
   + link_4,8 + link_6,7 + link_6,8 + link_7,8 <= 4
 s: link_1,5 + link_1,6 + link_1,7 + link_1,8 + link_5,6 + link_5,7
   + link_5,8 + link_6,7 + link_6,8 + link_7,8 <= 4
 s: link_2,3 + link_2,4 + link_2,5 + link_2,6 + link_3,4 + link_3,5
   + link_3,6 + link_4,5 + link_4,6 + link_5,6 <= 4
 s: link_2,3 + link_2,4 + link_2,5 + link_2,7 + link_3,4 + link_3,5
   + link_3,7 + link_4,5 + link_4,7 + link_5,7 <= 4
 s: link_2,3 + link_2,4 + link_2,5 + link_2,8 + link_3,4 + link_3,5
   + link_3,8 + link_4,5 + link_4,8 + link_5,8 <= 4
 s: link_2,3 + link_2,4 + link_2,6 + link_2,7 + link_3,4 + link_3,6
   + link_3,7 + link_4,6 + link_4,7 + link_6,7 <= 4
 s: link_2,3 + link_2,4 + link_2,6 + link_2,8 + link_3,4 + link_3,6
   + link_3,8 + link_4,6 + link_4,8 + link_6,8 <= 4
 s: link_2,3 + link_2,4 + link_2,7 + link_2,8 + link_3,4 + link_3,7
   + link_3,8 + link_4,7 + link_4,8 + link_7,8 <= 4
 s: link_2,3 + link_2,5 + link_2,6 + link_2,7 + link_3,5 + link_3,6
   + link_3,7 + link_5,6 + link_5,7 + link_6,7 <= 4
 s: link_2,3 + link_2,5 + link_2,6 + link_2,8 + link_3,5 + link_3,6
   + link_3,8 + link_5,6 + link_5,8 + link_6,8 <= 4
 s: link_2,3 + link_2,5 + link_2,7 + link_2,8 + link_3,5 + link_3,7
   + link_3,8 + link_5,7 + link_5,8 + link_7,8 <= 4
 s: link_2,3 + link_2,6 + link_2,7 + link_2,8 + link_3,6 + link_3,7
   + link_3,8 + link_6,7 + link_6,8 + link_7,8 <= 4
 s: link_2,4 + link_2,5 + link_2,6 + link_2,7 + link_4,5 + link_4,6
   + link_4,7 + link_5,6 + link_5,7 + link_6,7 <= 4
 s: link_2,4 + link_2,5 + link_2,6 + link_2,8 + link_4,5 + link_4,6
   + link_4,8 + link_5,6 + link_5,8 + link_6,8 <= 4
 s: link_2,4 + link_2,5 + link_2,7 + link_2,8 + link_4,5 + link_4,7
   + link_4,8 + link_5,7 + link_5,8 + link_7,8 <= 4
 s: link_2,4 + link_2,6 + link_2,7 + link_2,8 + link_4,6 + link_4,7
   + link_4,8 + link_6,7 + link_6,8 + link_7,8 <= 4
 s: link_2,5 + link_2,6 + link_2,7 + link_2,8 + link_5,6 + link_5,7
   + link_5,8 + link_6,7 + link_6,8 + link_7,8 <= 4
 s: link_3,4 + link_3,5 + link_3,6 + link_3,7 + link_4,5 + link_4,6
   + link_4,7 + link_5,6 + link_5,7 + link_6,7 <= 4
 s: link_3,4 + link_3,5 + link_3,6 + link_3,8 + link_4,5 + link_4,6
   + link_4,8 + link_5,6 + link_5,8 + link_6,8 <= 4
 s: link_3,4 + link_3,5 + link_3,7 + link_3,8 + link_4,5 + link_4,7
   + link_4,8 + link_5,7 + link_5,8 + link_7,8 <= 4
 s: link_3,4 + link_3,6 + link_3,7 + link_3,8 + link_4,6 + link_4,7
   + link_4,8 + link_6,7 + link_6,8 + link_7,8 <= 4
 s: link_3,5 + link_3,6 + link_3,7 + link_3,8 + link_5,6 + link_5,7
   + link_5,8 + link_6,7 + link_6,8 + link_7,8 <= 4
 s: link_4,5 + link_4,6 + link_4,7 + link_4,8 + link_5,6 + link_5,7
   + link_5,8 + link_6,7 + link_6,8 + link_7,8 <= 4
 s: link_1,2 + link_1,3 + link_1,4 + link_1,5 + link_1,6 + link_2,3
   + link_2,4 + link_2,5 + link_2,6 + link_3,4 + link_3,5 + link_3,6
   + link_4,5 + link_4,6 + link_5,6 <= 5
 s: link_1,2 + link_1,3 + link_1,4 + link_1,5 + link_1,7 + link_2,3
   + link_2,4 + link_2,5 + link_2,7 + link_3,4 + link_3,5 + link_3,7
   + link_4,5 + link_4,7 + link_5,7 <= 5
 s: link_1,2 + link_1,3 + link_1,4 + link_1,5 + link_1,8 + link_2,3
   + link_2,4 + link_2,5 + link_2,8 + link_3,4 + link_3,5 + link_3,8
   + link_4,5 + link_4,8 + link_5,8 <= 5
 s: link_1,2 + link_1,3 + link_1,4 + link_1,6 + link_1,7 + link_2,3
   + link_2,4 + link_2,6 + link_2,7 + link_3,4 + link_3,6 + link_3,7
   + link_4,6 + link_4,7 + link_6,7 <= 5
 s: link_1,2 + link_1,3 + link_1,4 + link_1,6 + link_1,8 + link_2,3
   + link_2,4 + link_2,6 + link_2,8 + link_3,4 + link_3,6 + link_3,8
   + link_4,6 + link_4,8 + link_6,8 <= 5
 s: link_1,2 + link_1,3 + link_1,4 + link_1,7 + link_1,8 + link_2,3
   + link_2,4 + link_2,7 + link_2,8 + link_3,4 + link_3,7 + link_3,8
   + link_4,7 + link_4,8 + link_7,8 <= 5
 s: link_1,2 + link_1,3 + link_1,5 + link_1,6 + link_1,7 + link_2,3
   + link_2,5 + link_2,6 + link_2,7 + link_3,5 + link_3,6 + link_3,7
   + link_5,6 + link_5,7 + link_6,7 <= 5
 s: link_1,2 + link_1,3 + link_1,5 + link_1,6 + link_1,8 + link_2,3
   + link_2,5 + link_2,6 + link_2,8 + link_3,5 + link_3,6 + link_3,8
   + link_5,6 + link_5,8 + link_6,8 <= 5
 s: link_1,2 + link_1,3 + link_1,5 + link_1,7 + link_1,8 + link_2,3
   + link_2,5 + link_2,7 + link_2,8 + link_3,5 + link_3,7 + link_3,8
   + link_5,7 + link_5,8 + link_7,8 <= 5
 s: link_1,2 + link_1,3 + link_1,6 + link_1,7 + link_1,8 + link_2,3
   + link_2,6 + link_2,7 + link_2,8 + link_3,6 + link_3,7 + link_3,8
   + link_6,7 + link_6,8 + link_7,8 <= 5
 s: link_1,2 + link_1,4 + link_1,5 + link_1,6 + link_1,7 + link_2,4
   + link_2,5 + link_2,6 + link_2,7 + link_4,5 + link_4,6 + link_4,7
   + link_5,6 + link_5,7 + link_6,7 <= 5
 s: link_1,2 + link_1,4 + link_1,5 + link_1,6 + link_1,8 + link_2,4
   + link_2,5 + link_2,6 + link_2,8 + link_4,5 + link_4,6 + link_4,8
   + link_5,6 + link_5,8 + link_6,8 <= 5
 s: link_1,2 + link_1,4 + link_1,5 + link_1,7 + link_1,8 + link_2,4
   + link_2,5 + link_2,7 + link_2,8 + link_4,5 + link_4,7 + link_4,8
   + link_5,7 + link_5,8 + link_7,8 <= 5
 s: link_1,2 + link_1,4 + link_1,6 + link_1,7 + link_1,8 + link_2,4
   + link_2,6 + link_2,7 + link_2,8 + link_4,6 + link_4,7 + link_4,8
   + link_6,7 + link_6,8 + link_7,8 <= 5
 s: link_1,2 + link_1,5 + link_1,6 + link_1,7 + link_1,8 + link_2,5
   + link_2,6 + link_2,7 + link_2,8 + link_5,6 + link_5,7 + link_5,8
   + link_6,7 + link_6,8 + link_7,8 <= 5
 s: link_1,3 + link_1,4 + link_1,5 + link_1,6 + link_1,7 + link_3,4
   + link_3,5 + link_3,6 + link_3,7 + link_4,5 + link_4,6 + link_4,7
   + link_5,6 + link_5,7 + link_6,7 <= 5
 s: link_1,3 + link_1,4 + link_1,5 + link_1,6 + link_1,8 + link_3,4
   + link_3,5 + link_3,6 + link_3,8 + link_4,5 + link_4,6 + link_4,8
   + link_5,6 + link_5,8 + link_6,8 <= 5
 s: link_1,3 + link_1,4 + link_1,5 + link_1,7 + link_1,8 + link_3,4
   + link_3,5 + link_3,7 + link_3,8 + link_4,5 + link_4,7 + link_4,8
   + link_5,7 + link_5,8 + link_7,8 <= 5
 s: link_1,3 + link_1,4 + link_1,6 + link_1,7 + link_1,8 + link_3,4
   + link_3,6 + link_3,7 + link_3,8 + link_4,6 + link_4,7 + link_4,8
   + link_6,7 + link_6,8 + link_7,8 <= 5
 s: link_1,3 + link_1,5 + link_1,6 + link_1,7 + link_1,8 + link_3,5
   + link_3,6 + link_3,7 + link_3,8 + link_5,6 + link_5,7 + link_5,8
   + link_6,7 + link_6,8 + link_7,8 <= 5
 s: link_1,4 + link_1,5 + link_1,6 + link_1,7 + link_1,8 + link_4,5
   + link_4,6 + link_4,7 + link_4,8 + link_5,6 + link_5,7 + link_5,8
   + link_6,7 + link_6,8 + link_7,8 <= 5
 s: link_2,3 + link_2,4 + link_2,5 + link_2,6 + link_2,7 + link_3,4
   + link_3,5 + link_3,6 + link_3,7 + link_4,5 + link_4,6 + link_4,7
   + link_5,6 + link_5,7 + link_6,7 <= 5
 s: link_2,3 + link_2,4 + link_2,5 + link_2,6 + link_2,8 + link_3,4
   + link_3,5 + link_3,6 + link_3,8 + link_4,5 + link_4,6 + link_4,8
   + link_5,6 + link_5,8 + link_6,8 <= 5
 s: link_2,3 + link_2,4 + link_2,5 + link_2,7 + link_2,8 + link_3,4
   + link_3,5 + link_3,7 + link_3,8 + link_4,5 + link_4,7 + link_4,8
   + link_5,7 + link_5,8 + link_7,8 <= 5
 s: link_2,3 + link_2,4 + link_2,6 + link_2,7 + link_2,8 + link_3,4
   + link_3,6 + link_3,7 + link_3,8 + link_4,6 + link_4,7 + link_4,8
   + link_6,7 + link_6,8 + link_7,8 <= 5
 s: link_2,3 + link_2,5 + link_2,6 + link_2,7 + link_2,8 + link_3,5
   + link_3,6 + link_3,7 + link_3,8 + link_5,6 + link_5,7 + link_5,8
   + link_6,7 + link_6,8 + link_7,8 <= 5
 s: link_2,4 + link_2,5 + link_2,6 + link_2,7 + link_2,8 + link_4,5
   + link_4,6 + link_4,7 + link_4,8 + link_5,6 + link_5,7 + link_5,8
   + link_6,7 + link_6,8 + link_7,8 <= 5
 s: link_3,4 + link_3,5 + link_3,6 + link_3,7 + link_3,8 + link_4,5
   + link_4,6 + link_4,7 + link_4,8 + link_5,6 + link_5,7 + link_5,8
   + link_6,7 + link_6,8 + link_7,8 <= 5
 s: link_1,2 + link_1,3 + link_1,4 + link_1,5 + link_1,6 + link_1,7
   + link_2,3 + link_2,4 + link_2,5 + link_2,6 + link_2,7 + link_3,4
   + link_3,5 + link_3,6 + link_3,7 + link_4,5 + link_4,6 + link_4,7
   + link_5,6 + link_5,7 + link_6,7 <= 6
 s: link_1,2 + link_1,3 + link_1,4 + link_1,5 + link_1,6 + link_1,8
   + link_2,3 + link_2,4 + link_2,5 + link_2,6 + link_2,8 + link_3,4
   + link_3,5 + link_3,6 + link_3,8 + link_4,5 + link_4,6 + link_4,8
   + link_5,6 + link_5,8 + link_6,8 <= 6
 s: link_1,2 + link_1,3 + link_1,4 + link_1,5 + link_1,7 + link_1,8
   + link_2,3 + link_2,4 + link_2,5 + link_2,7 + link_2,8 + link_3,4
   + link_3,5 + link_3,7 + link_3,8 + link_4,5 + link_4,7 + link_4,8
   + link_5,7 + link_5,8 + link_7,8 <= 6
 s: link_1,2 + link_1,3 + link_1,4 + link_1,6 + link_1,7 + link_1,8
   + link_2,3 + link_2,4 + link_2,6 + link_2,7 + link_2,8 + link_3,4
   + link_3,6 + link_3,7 + link_3,8 + link_4,6 + link_4,7 + link_4,8
   + link_6,7 + link_6,8 + link_7,8 <= 6
 s: link_1,2 + link_1,3 + link_1,5 + link_1,6 + link_1,7 + link_1,8
   + link_2,3 + link_2,5 + link_2,6 + link_2,7 + link_2,8 + link_3,5
   + link_3,6 + link_3,7 + link_3,8 + link_5,6 + link_5,7 + link_5,8
   + link_6,7 + link_6,8 + link_7,8 <= 6
 s: link_1,2 + link_1,4 + link_1,5 + link_1,6 + link_1,7 + link_1,8
   + link_2,4 + link_2,5 + link_2,6 + link_2,7 + link_2,8 + link_4,5
   + link_4,6 + link_4,7 + link_4,8 + link_5,6 + link_5,7 + link_5,8
   + link_6,7 + link_6,8 + link_7,8 <= 6
 s: link_1,3 + link_1,4 + link_1,5 + link_1,6 + link_1,7 + link_1,8
   + link_3,4 + link_3,5 + link_3,6 + link_3,7 + link_3,8 + link_4,5
   + link_4,6 + link_4,7 + link_4,8 + link_5,6 + link_5,7 + link_5,8
   + link_6,7 + link_6,8 + link_7,8 <= 6
 s: link_2,3 + link_2,4 + link_2,5 + link_2,6 + link_2,7 + link_2,8
   + link_3,4 + link_3,5 + link_3,6 + link_3,7 + link_3,8 + link_4,5
   + link_4,6 + link_4,7 + link_4,8 + link_5,6 + link_5,7 + link_5,8
   + link_6,7 + link_6,8 + link_7,8 <= 6
Bounds
Binaries
 link_1,2 link_1,3 link_1,4 link_1,5 link_1,6 link_1,7 link_1,8 link_2,3
 link_2,4 link_2,5 link_2,6 link_2,7 link_2,8 link_3,4 link_3,5 link_3,6
 link_3,7 link_3,8 link_4,5 link_4,6 link_4,7 link_4,8 link_5,6 link_5,7
 link_5,8 link_6,7 link_6,8 link_7,8
End
