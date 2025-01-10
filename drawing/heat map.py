import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
matrix_FedMatch_0 = np.array([[1.0000001, 0.9970201, 0.5051704, 0.4745019, 0.7410601, 0.69422424, 0.63426435, 0.6830073, 0.6466469, 0.70515656],
 [0.9970201, 1, 0.4915283, 0.46204594, 0.74569607, 0.7006304, 0.5966097, 0.6432084, 0.6397656, 0.69867176],
 [0.5051704, 0.4915283, 1, 0.99856585, 0.6636847, 0.595451, 0.5136714, 0.4738974, 0.6781254, 0.67785573],
 [0.4745019, 0.46204594, 0.99856585, 0.99999994, 0.6483815, 0.57970876, 0.48310155, 0.44029012, 0.6786121, 0.6755057 ],
 [0.7410601, 0.74569607, 0.6636847, 0.6483815, 1.0000001, 0.99259096, 0.43259043, 0.49148265, 0.82149434, 0.8513551 ],
 [0.69422424, 0.7006304, 0.595451, 0.57970876, 0.99259096, 1, 0.3590781, 0.4282183, 0.7935897, 0.82185966],
 [0.63426435, 0.5966097, 0.5136714, 0.48310155, 0.43259043, 0.3590781, 1.0000001, 0.9886463, 0.3712221, 0.39480114],
 [0.6830073, 0.6432084, 0.4738974, 0.44029012, 0.49148265, 0.4282183, 0.9886463, 1, 0.41057265, 0.44069427],
 [0.6466469, 0.6397656, 0.6781254, 0.6786121, 0.82149434, 0.7935897, 0.3712221, 0.41057265, 1, 0.99525934],
 [0.70515656, 0.69867176, 0.67785573, 0.6755057, 0.8513551, 0.82185966, 0.39480114, 0.44069427, 0.99525934, 1.0000001 ]]
)
matrix_FedMatch_1 = ( [[ 1, 0.9948795, 0.4259412, 0.5018377, 0.17568538, 0.21040195,
  -0.2544996,  -0.2752375,  -0.18959023, -0.1695054 ],
 [ 0.9948795, 0.99999994, 0.44711003, 0.5385529, 0.1897905, 0.22894014,
  -0.23376729, -0.25041717, -0.15233916, -0.12760866],
 [ 0.4259412, 0.44711003, 1.0000001, 0.955837, 0.4198116, 0.42215884, 0.3678018, 0.37553313, 0.0097672, 0.05765752],
 [ 0.5018377, 0.5385529, 0.955837, 1, 0.4762442, 0.50051814, 0.24256183, 0.25707787, 0.1322347, 0.18651374],
 [ 0.17568538, 0.1897905, 0.4198116, 0.4762442, 1.0000001, 0.995942, 0.07199156, 0.12487538, -0.01774312, 0.01023588],
 [ 0.21040195, 0.22894014, 0.42215884, 0.50051814, 0.995942, 0.99999994, 0.06228681, 0.11478674, 0.02044821, 0.052579  ],
 [-0.2544996,  -0.23376729, 0.3678018, 0.24256183, 0.07199156, 0.06228681, 1, 0.99655527, -0.18237509, -0.12969705],
 [-0.2752375,  -0.25041717, 0.37553313, 0.25707787, 0.12487538, 0.11478674, 0.99655527, 1,-0.14037603, -0.08780236],
 [-0.18959023, -0.15233916, 0.0097672, 0.1322347,  -0.01774312, 0.02044821,
  -0.18237509, -0.14037603, 0.99999994, 0.9963903 ],
 [-0.1695054,  -0.12760866, 0.05765752, 0.18651374, 0.01023588, 0.052579,
  -0.12969705, -0.08780236, 0.9963903, 0.9999999 ]])
matrix_FedMatch_2 = ([[ 0.99999994, 0.9997159, 0.48625004, 0.49292493, 0.38781407, 0.3873953, -0.10638598, -0.11231025, 0.10642518, 0.12234238],
 [ 0.9997159, 1.0000001, 0.4899171, 0.49651948, 0.39627433, 0.3960651, -0.10125187, -0.10730356, 0.1127817, 0.12882169],
 [ 0.48625004, 0.4899171, 1., 0.9992921, 0.55523664, 0.5472433, 0.46794537, 0.46104074, 0.4267626, 0.4395482, ],
 [ 0.49292493, 0.49651948, 0.9992921, 1.0000001, 0.54296994, 0.53495395, 0.4722706, 0.46523523, 0.4013388, 0.41413796],
 [ 0.38781407, 0.39627433, 0.55523664, 0.54296994, 0.99999994, 0.9998108, 0.20953417, 0.20043847, 0.20899455, 0.2268709, ],
 [ 0.3873953, 0.3960651, 0.5472433, 0.53495395, 0.9998108, 0.9999999, 0.2035509, 0.1943786, 0.20877479, 0.22666365],
 [-0.10638598, -0.10125187, 0.46794537, 0.4722706, 0.20953417, 0.2035509, 1.0000001, 0.99989676, 0.08013446, 0.0865553, ],
 [-0.11231025, -0.10730356, 0.46104074, 0.46523523, 0.20043847, 0.1943786, 0.99989676, 1., 0.08250917, 0.08869993],
 [ 0.10642518, 0.1127817, 0.4267626, 0.4013388, 0.20899455, 0.20877479, 0.08013446, 0.08250917, 0.99999994, 0.9997198, ],
 [ 0.12234238, 0.12882169, 0.4395482, 0.41413796, 0.2268709, 0.22666365, 0.0865553, 0.08869993, 0.9997198, 1.0000001, ]]
)
matrix_FedAC_0 =([[ 1.0000,  0.9345,  0.0246,  0.0326, -0.0878, -0.1235,  0.1518,  0.1095,
         -0.1691, -0.1314],
        [ 0.9345,  1.0000, -0.0047,  0.0038, -0.0904, -0.1263,  0.0701,  0.0214,
         -0.1772, -0.1250],
        [ 0.0246, -0.0047,  1.0000,  0.9675, -0.0451, -0.1196,  0.1471,  0.1041,
          0.0940,  0.1223],
        [ 0.0326,  0.0038,  0.9675,  1.0000, -0.0612, -0.1362,  0.1616,  0.1224,
          0.1155,  0.1397],
        [-0.0878, -0.0904, -0.0451, -0.0612,  1.0000,  0.9114, -0.2029, -0.1743,
         -0.2035, -0.1946],
        [-0.1235, -0.1263, -0.1196, -0.1362,  0.9114,  1.0000, -0.2454, -0.2097,
         -0.1898, -0.1818],
        [ 0.1518,  0.0701,  0.1471,  0.1616, -0.2029, -0.2454,  1.0000,  0.9494,
         -0.0530, -0.0407],
        [ 0.1095,  0.0214,  0.1041,  0.1224, -0.1743, -0.2097,  0.9494,  1.0000,
         -0.0700, -0.0653],
        [-0.1691, -0.1772,  0.0940,  0.1155, -0.2035, -0.1898, -0.0530, -0.0700,
          1.0000,  0.9268],
        [-0.1314, -0.1250,  0.1223,  0.1397, -0.1946, -0.1818, -0.0407, -0.0653,
          0.9268,  1.0000]])
matrix_FedAC_1 = ([[1.0000, 0.2652, 0.1339, 0.0802, 0.1268, 0.1392, 0.1212, 0.1346, 0.1359,
         0.1071],
        [0.2652, 1.0000, 0.1318, 0.0737, 0.1218, 0.1244, 0.1532, 0.1864, 0.1088,
         0.1073],
        [0.1339, 0.1318, 1.0000, 0.2555, 0.1014, 0.1297, 0.1136, 0.1430, 0.1459,
         0.2002],
        [0.0802, 0.0737, 0.2555, 1.0000, 0.0740, 0.0987, 0.0742, 0.1077, 0.1150,
         0.1400],
        [0.1268, 0.1218, 0.1014, 0.0740, 1.0000, 0.2464, 0.1319, 0.1448, 0.1095,
         0.1220],
        [0.1392, 0.1244, 0.1297, 0.0987, 0.2464, 1.0000, 0.0924, 0.1303, 0.0908,
         0.1415],
        [0.1212, 0.1532, 0.1136, 0.0742, 0.1319, 0.0924, 1.0000, 0.3263, 0.1077,
         0.1085],
        [0.1346, 0.1864, 0.1430, 0.1077, 0.1448, 0.1303, 0.3263, 1.0000, 0.1256,
         0.1212],
        [0.1359, 0.1088, 0.1459, 0.1150, 0.1095, 0.0908, 0.1077, 0.1256, 1.0000,
         0.2643],
        [0.1071, 0.1073, 0.2002, 0.1400, 0.1220, 0.1415, 0.1085, 0.1212, 0.2643,
         1.0000]])
matrix_FedAC_2 = ([[ 1.0000,  0.1054,  0.0855,  0.0683,  0.1401,  0.2773,  0.1815,  0.2004,
          0.1533,  0.2088],
        [ 0.1054,  1.0000, -0.0165,  0.1148,  0.1373,  0.1159,  0.1351,  0.1307,
          0.0389,  0.0475],
        [ 0.0855, -0.0165,  1.0000, -0.1044,  0.1619, -0.0395,  0.0647,  0.0370,
          0.3779,  0.3580],
        [ 0.0683,  0.1148, -0.1044,  1.0000,  0.1073,  0.2076,  0.2141,  0.2695,
          0.0704,  0.0609],
        [ 0.1401,  0.1373,  0.1619,  0.1073,  1.0000,  0.2008,  0.1421,  0.1812,
          0.2973,  0.3206],
        [ 0.2773,  0.1159, -0.0395,  0.2076,  0.2008,  1.0000,  0.1785,  0.2661,
          0.1267,  0.1514],
        [ 0.1815,  0.1351,  0.0647,  0.2141,  0.1421,  0.1785,  1.0000,  0.6662,
          0.1001,  0.1224],
        [ 0.2004,  0.1307,  0.0370,  0.2695,  0.1812,  0.2661,  0.6662,  1.0000,
          0.0886,  0.1245],
        [ 0.1533,  0.0389,  0.3779,  0.0704,  0.2973,  0.1267,  0.1001,  0.0886,
          1.0000,  0.8379],
        [ 0.2088,  0.0475,  0.3580,  0.0609,  0.3206,  0.1514,  0.1224,  0.1245,
          0.8379,  1.0000]])
matrix_FedCAC_0 = ([[1.00000000, 0.86153829, 0.55128116, 0.55104649, 0.55446380, 0.55511547,
  0.55681938, 0.55762381, 0.55221159, 0.55081898],
 [0.86153829, 1.00000000, 0.55029158, 0.55000000, 0.55285470, 0.55404500,
  0.55548592, 0.55622856, 0.55205999, 0.55060794],
 [0.55128116, 0.55029158, 1.00000000, 0.90161163, 0.55420700, 0.55494834,
  0.55790633, 0.55751875, 0.55564165, 0.55487742],
 [0.55104649, 0.55000000, 0.90161163, 1.00000000, 0.55373796, 0.55485055,
  0.55760030, 0.55707788, 0.55660183, 0.55596515],
 [0.55446380, 0.55285470, 0.55420700, 0.55373796, 1.00000000, 0.85658371,
  0.55121717, 0.55189704, 0.55681309, 0.55548145],
 [0.55511547, 0.55404500, 0.55494834, 0.55485055, 0.85658371, 1.00000000,
  0.55186382, 0.55194290, 0.55700905, 0.55555303],
 [0.55681938, 0.55548592, 0.55790633, 0.55760030, 0.55121717, 0.55186382,
  1.00000000, 0.87333572, 0.55284744, 0.55156769],
 [0.55762381, 0.55622856, 0.55751875, 0.55707788, 0.55189704, 0.55194290,
  0.87333572, 1.00000000, 0.55278691, 0.55153685],
 [0.55221159, 0.55205999, 0.55564165, 0.55660183, 0.55681309, 0.55700905,
  0.55284744, 0.55278691, 1.00000000, 0.86054063],
 [0.55081898, 0.55060794, 0.55487742, 0.55596515, 0.55548145, 0.55555303,
  0.55156769, 0.55153685, 0.86054063, 1.00000000]]
)
matrix_FedCAC_1 = ([[1.00000000, 0.56763261, 0.55451709, 0.55300123, 0.55411720, 0.55336230,
  0.55258015, 0.55308699, 0.55371276, 0.55308762],
 [0.56763261, 1.00000000, 0.55720128, 0.55569100, 0.55647710, 0.55540876,
  0.55579513, 0.55654139, 0.55630745, 0.55523581],
 [0.55451709, 0.55720128, 1.00000000, 0.56956181, 0.55439102, 0.55400117,
  0.55473914, 0.55470413, 0.55772286, 0.55780208],
 [0.55300123, 0.55569100, 0.56956181, 1.00000000, 0.55220889, 0.55170161,
  0.55311228, 0.55343837, 0.55644743, 0.55639162],
 [0.55411720, 0.55647710, 0.55439102, 0.55220889, 1.00000000, 0.56615233,
  0.55096625, 0.55213441, 0.55452788, 0.55416873],
 [0.55336230, 0.55540876, 0.55400117, 0.55170161, 0.56615233, 1.00000000,
  0.55000000, 0.55123557, 0.55416357, 0.55387366],
 [0.55258015, 0.55579513, 0.55473914, 0.55311228, 0.55096625, 0.55000000,
  1.00000000, 0.56816483, 0.55156987, 0.55123623],
 [0.55308699, 0.55654139, 0.55470413, 0.55343837, 0.55213441, 0.55123557,
  0.56816483, 1.00000000, 0.55264936, 0.55240525],
 [0.55371276, 0.55630745, 0.55772286, 0.55644743, 0.55452788, 0.55416357,
  0.55156987, 0.55264936, 1.00000000, 0.57042012],
 [0.55308762, 0.55523581, 0.55780208, 0.55639162, 0.55416873, 0.55387366,
  0.55123623, 0.55240525, 0.57042012, 1.00000000]])
matrix_FedCAC_2 = ([[1.00000000, 0.57356516, 0.55459137, 0.55411739, 0.55930773, 0.56178775,
  0.55962116, 0.55961406, 0.55441349, 0.55717910],
 [0.57356516, 1.00000000, 0.55253241, 0.55282766, 0.56024503, 0.56200776,
  0.55804068, 0.55828175, 0.55360604, 0.55703604],
 [0.55459137, 0.55253241, 1.00000000, 0.57168047, 0.55437941, 0.55409758,
  0.55414974, 0.55303290, 0.56264923, 0.56067751],
 [0.55411739, 0.55282766, 0.57168047, 1.00000000, 0.55471000, 0.55659327,
  0.55506954, 0.55628734, 0.55794711, 0.55621070],
 [0.55930773, 0.56024503, 0.55437941, 0.55471000, 1.00000000, 0.57286015,
  0.55380304, 0.55347023, 0.56001631, 0.56266701],
 [0.56178775, 0.56200776, 0.55409758, 0.55659327, 0.57286015, 1.00000000,
  0.55521355, 0.55628792, 0.55898178, 0.56177195],
 [0.55962116, 0.55804068, 0.55414974, 0.55506954, 0.55380304, 0.55521355,
  1.00000000, 0.57942488, 0.55000000, 0.55196876],
 [0.55961406, 0.55828175, 0.55303290, 0.55628734, 0.55347023, 0.55628792,
  0.57942488, 1.00000000, 0.55007933, 0.55226272],
 [0.55441349, 0.55360604, 0.56264923, 0.55794711, 0.56001631, 0.55898178,
  0.55000000, 0.55007933, 1.00000000, 0.59479951],
 [0.55717910, 0.55703604, 0.56067751, 0.55621070, 0.56266701, 0.56177195,
  0.55196876, 0.55226272, 0.59479951, 1.00000000]])
matrix_pFSSLD_0 = ([[1., 0.99565578, 0.36700922, 0.35959595, 0.47424611, 0.35454261, 0.56694615, 0.53633606, 0.42406827, 0.49164587],
 [0.99565578, 1., 0.32533142, 0.3177197, 0.44419482, 0.32616967, 0.52340311, 0.48987472, 0.39296019, 0.46199077],
 [0.36700922, 0.32533142, 1., 0.99965417, 0.32053739, 0.21958379, 0.46882278, 0.44989064, 0.26819077, 0.31482881],
 [0.35959595, 0.3177197, 0.99965417, 1., 0.31335205, 0.21299949, 0.46218067, 0.44416621, 0.25877789, 0.30581534],
 [0.47424611, 0.44419482, 0.32053739, 0.31335205, 1., 0.98316699, 0.45600152, 0.46390665, 0.42102796, 0.46461999],
 [0.35454261, 0.32616967, 0.21958379, 0.21299949, 0.98316699, 1., 0.34773046, 0.36179951, 0.36053297, 0.39300799],
 [0.56694615, 0.52340311, 0.46882278, 0.46218067, 0.45600152, 0.34773046, 1., 0.99516195, 0.40048483, 0.45689827],
 [0.53633606, 0.48987472, 0.44989064, 0.44416621, 0.46390665, 0.36179951, 0.99516195, 1., 0.39210761, 0.4431048, ],
 [0.42406827, 0.39296019, 0.26819077, 0.25877789, 0.42102796, 0.36053297, 0.40048483, 0.39210761, 1., 0.9835524, ],
 [0.49164587, 0.46199077, 0.31482881, 0.30581534, 0.46461999, 0.39300799, 0.45689827, 0.4431048, 0.9835524, 1., ]])
matrix_pFSSLD_1 = ([[1., 0.9979564, 0.21720006, 0.23129123, 0.12129165, 0.10386211, 0.030276, 0.04873592, 0.05241024, 0.04928504],
 [0.9979564, 1., 0.19156355, 0.20325901, 0.11469147, 0.09670404, 0.03964972, 0.06064654, 0.07238889, 0.0677857, ],
 [0.21720006, 0.19156355, 1., 0.99581504, 0.18646213, 0.14569138, 0.1410885, 0.22294189, 0.00724808, 0.00806201],
 [0.23129123, 0.20325901, 0.99581504, 1., 0.19325693, 0.15168139, 0.1317765, 0.20741545, 0.01242563, 0.01405358],
 [0.12129165, 0.11469147, 0.18646213, 0.19325693, 1., 0.99508387, 0.13647312, 0.20051795, 0.11069179, 0.100353, ],
 [0.10386211, 0.09670404, 0.14569138, 0.15168139, 0.99508387, 1., 0.11381632, 0.17379566, 0.06964287, 0.06360454],
 [0.030276, 0.03964972, 0.1410885, 0.1317765, 0.13647312, 0.11381632, 1., 0.9726032, 0.05538578, 0.06074475],
 [0.04873592, 0.06064654, 0.22294189, 0.20741545, 0.20051795, 0.17379566, 0.9726032, 1., 0.09309079, 0.09915501],
 [0.05241024, 0.07238889, 0.00724808, 0.01242563, 0.11069179, 0.06964287, 0.05538578, 0.09309079, 1., 0.99254507],
 [0.04928504, 0.0677857, 0.00806201, 0.01405358, 0.100353, 0.06360454, 0.06074475, 0.09915501, 0.99254507, 1., ]])
matrix_pFSSLD_2 = ([[1., 0.99896073, 0.33056045, 0.33440155, 0.35943502, 0.35398358, 0.23861346, 0.24039952, 0.26438665, 0.27135211],
 [0.99896073, 1., 0.33692113, 0.34052211, 0.37318772, 0.36780852, 0.24699628, 0.24884889, 0.27416253, 0.28128824],
 [0.33056045, 0.33692113, 1., 0.99791467, 0.37670159, 0.36441457, 0.34692591, 0.35033742, 0.10861595, 0.11297567],
 [0.33440155, 0.34052211, 0.99791467, 1., 0.3743962, 0.36232087, 0.33431366, 0.33738202, 0.10589139, 0.11023894],
 [0.35943502, 0.37318772, 0.37670159, 0.3743962, 1., 0.99869215, 0.45067805, 0.45233971, 0.28732312, 0.29542792],
 [0.35398358, 0.36780852, 0.36441457, 0.36232087, 0.99869215, 1., 0.43957564, 0.44108891, 0.27915627, 0.28733832],
 [0.23861346, 0.24699628, 0.34692591, 0.33431366, 0.45067805, 0.43957564, 1., 0.99995458, 0.30717492, 0.31501567],
 [0.24039952, 0.24884889, 0.35033742, 0.33738202, 0.45233971, 0.44108891, 0.99995458, 1., 0.31008875, 0.3179287, ],
 [0.26438665, 0.27416253, 0.10861595, 0.10589139, 0.28732312, 0.27915627, 0.30717492, 0.31008875, 1., 0.99993223],
 [0.27135211, 0.28128824, 0.11297567, 0.11023894, 0.29542792, 0.28733832, 0.31501567, 0.3179287, 0.99993223, 1., ]])
matrix_ideal = ([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
       [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
       [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]])
matrix = np.array(matrix_FedCAC_2)


cmap = LinearSegmentedColormap.from_list('viridis_custom', [
    (0.0, '#440154'),   # Viridis 深紫色
    (0.30, '#3F1A62'),   # Viridis 深蓝色
    (0.33, '#3A3370'),
    (0.55, '#31688e'),  # Viridis 蓝色
    (0.66, '#35b779'),  # Viridis 绿色
    (1.0, '#fde725')    # Viridis 黄色
], N=256)
# 绘制热力图
plt.figure(figsize=(8, 7))
plt.imshow(matrix, cmap=cmap, interpolation='nearest', vmin=0, vmax=1)
colorbar = plt.colorbar()
# 调整色条刻度的字号
colorbar.ax.tick_params(labelsize=20)  # 将色条刻度字号设置为15
# 添加标题和轴标签
plt.title('Estimated Similarity at Epoch 50', fontsize=22)
plt.xlabel('Client Index', fontsize=22)
plt.ylabel('Client Index', fontsize=22)

# 设置刻度，使得每个 Client Index 都显示，并调整字号
num_clients = matrix.shape[0]  # 获取矩阵的大小
plt.xticks(np.arange(num_clients), fontsize=20)  # 调整 X 轴刻度的字号
plt.yticks(np.arange(num_clients), fontsize=20)  # 调整 Y 轴刻度的字号

# 保存和显示图像
plt.savefig("FedCAC_2.jpg", dpi=300)
plt.show()