import matplotlib.pyplot as plt
import re

# Assume the text data is stored in a variable named text
text = """
Anatomical Region Classification Metrics:
2025-03-23 17:05:49,829 - 
Region 0:
2025-03-23 17:05:49,829 - Sample Level Precision: 0.4701
2025-03-23 17:05:49,829 - Sample Level Recall: 0.4026
2025-03-23 17:05:49,829 - Sample Level F1 Score: 0.4164
2025-03-23 17:05:49,829 - Class Level Precision: 0.3303
2025-03-23 17:05:49,829 - Class Level Recall: 0.1756
2025-03-23 17:05:49,829 - Class Level F1 Score: 0.2102
2025-03-23 17:05:49,829 - 
Region 1:
2025-03-23 17:05:49,829 - Sample Level Precision: 0.3837
2025-03-23 17:05:49,829 - Sample Level Recall: 0.3278
2025-03-23 17:05:49,829 - Sample Level F1 Score: 0.3412
2025-03-23 17:05:49,830 - Class Level Precision: 0.3764
2025-03-23 17:05:49,830 - Class Level Recall: 0.0933
2025-03-23 17:05:49,830 - Class Level F1 Score: 0.1042
2025-03-23 17:05:49,830 - 
Region 2:
2025-03-23 17:05:49,830 - Sample Level Precision: 0.4736
2025-03-23 17:05:49,830 - Sample Level Recall: 0.4057
2025-03-23 17:05:49,830 - Sample Level F1 Score: 0.4198
2025-03-23 17:05:49,830 - Class Level Precision: 0.2969
2025-03-23 17:05:49,830 - Class Level Recall: 0.1777
2025-03-23 17:05:49,830 - Class Level F1 Score: 0.2098
2025-03-23 17:05:49,830 - 
Region 3:
2025-03-23 17:05:49,830 - Sample Level Precision: 0.4064
2025-03-23 17:05:49,830 - Sample Level Recall: 0.3411
2025-03-23 17:05:49,830 - Sample Level F1 Score: 0.3568
2025-03-23 17:05:49,830 - Class Level Precision: 0.2908
2025-03-23 17:05:49,830 - Class Level Recall: 0.1063
2025-03-23 17:05:49,830 - Class Level F1 Score: 0.1222
2025-03-23 17:05:49,830 - 
Region 4:
2025-03-23 17:05:49,830 - Sample Level Precision: 0.3697
2025-03-23 17:05:49,830 - Sample Level Recall: 0.3186
2025-03-23 17:05:49,830 - Sample Level F1 Score: 0.3306
2025-03-23 17:05:49,830 - Class Level Precision: 0.0915
2025-03-23 17:05:49,830 - Class Level Recall: 0.0827
2025-03-23 17:05:49,830 - Class Level F1 Score: 0.0868
2025-03-23 17:05:49,830 - 
Region 5:
2025-03-23 17:05:49,830 - Sample Level Precision: 0.3986
2025-03-23 17:05:49,830 - Sample Level Recall: 0.3384
2025-03-23 17:05:49,830 - Sample Level F1 Score: 0.3523
2025-03-23 17:05:49,830 - Class Level Precision: 0.3032
2025-03-23 17:05:49,830 - Class Level Recall: 0.1047
2025-03-23 17:05:49,830 - Class Level F1 Score: 0.1232
2025-03-23 17:05:49,830 - 
Region 6:
2025-03-23 17:05:49,830 - Sample Level Precision: 0.4398
2025-03-23 17:05:49,831 - Sample Level Recall: 0.3651
2025-03-23 17:05:49,831 - Sample Level F1 Score: 0.3820
2025-03-23 17:05:49,831 - Class Level Precision: 0.2482
2025-03-23 17:05:49,831 - Class Level Recall: 0.1363
2025-03-23 17:05:49,831 - Class Level F1 Score: 0.1599
2025-03-23 17:05:49,831 - 
Region 7:
2025-03-23 17:05:49,831 - Sample Level Precision: 0.4656
2025-03-23 17:05:49,831 - Sample Level Recall: 0.3956
2025-03-23 17:05:49,831 - Sample Level F1 Score: 0.4108
2025-03-23 17:05:49,831 - Class Level Precision: 0.2886
2025-03-23 17:05:49,831 - Class Level Recall: 0.1704
2025-03-23 17:05:49,831 - Class Level F1 Score: 0.1984
2025-03-23 17:05:49,831 - 
Region 8:
2025-03-23 17:05:49,831 - Sample Level Precision: 0.4422
2025-03-23 17:05:49,831 - Sample Level Recall: 0.3680
2025-03-23 17:05:49,831 - Sample Level F1 Score: 0.3847
2025-03-23 17:05:49,831 - Class Level Precision: 0.2555
2025-03-23 17:05:49,831 - Class Level Recall: 0.1395
2025-03-23 17:05:49,831 - Class Level F1 Score: 0.1636
2025-03-23 17:05:49,831 - 
Region 9:
2025-03-23 17:05:49,831 - Sample Level Precision: 0.4239
2025-03-23 17:05:49,831 - Sample Level Recall: 0.3598
2025-03-23 17:05:49,831 - Sample Level F1 Score: 0.3746
2025-03-23 17:05:49,831 - Class Level Precision: 0.2882
2025-03-23 17:05:49,831 - Class Level Recall: 0.1315
2025-03-23 17:05:49,831 - Class Level F1 Score: 0.1602
2025-03-23 17:05:49,831 - 
Region 10:
2025-03-23 17:05:49,831 - Sample Level Precision: 0.3801
2025-03-23 17:05:49,831 - Sample Level Recall: 0.3249
2025-03-23 17:05:49,831 - Sample Level F1 Score: 0.3379
2025-03-23 17:05:49,831 - Class Level Precision: 0.2159
2025-03-23 17:05:49,831 - Class Level Recall: 0.0920
2025-03-23 17:05:49,831 - Class Level F1 Score: 0.1025
2025-03-23 17:05:49,831 - 
Region 11:
2025-03-23 17:05:49,831 - Sample Level Precision: 0.4676
2025-03-23 17:05:49,831 - Sample Level Recall: 0.3985
2025-03-23 17:05:49,832 - Sample Level F1 Score: 0.4133
2025-03-23 17:05:49,832 - Class Level Precision: 0.2813
2025-03-23 17:05:49,832 - Class Level Recall: 0.1712
2025-03-23 17:05:49,832 - Class Level F1 Score: 0.1980
2025-03-23 17:05:49,832 - 
Region 12:
2025-03-23 17:05:49,832 - Sample Level Precision: 0.4399
2025-03-23 17:05:49,832 - Sample Level Recall: 0.3658
2025-03-23 17:05:49,832 - Sample Level F1 Score: 0.3825
2025-03-23 17:05:49,832 - Class Level Precision: 0.3149
2025-03-23 17:05:49,832 - Class Level Recall: 0.1377
2025-03-23 17:05:49,832 - Class Level F1 Score: 0.1609
2025-03-23 17:05:49,832 - 
Region 13:
2025-03-23 17:05:49,832 - Sample Level Precision: 0.3826
2025-03-23 17:05:49,832 - Sample Level Recall: 0.3260
2025-03-23 17:05:49,832 - Sample Level F1 Score: 0.3393
2025-03-23 17:05:49,832 - Class Level Precision: 0.2756
2025-03-23 17:05:49,832 - Class Level Recall: 0.0928
2025-03-23 17:05:49,832 - Class Level F1 Score: 0.1037
2025-03-23 17:05:49,832 - 
Region 14:
2025-03-23 17:05:49,832 - Sample Level Precision: 0.4099
2025-03-23 17:05:49,832 - Sample Level Recall: 0.3505
2025-03-23 17:05:49,832 - Sample Level F1 Score: 0.3642
2025-03-23 17:05:49,832 - Class Level Precision: 0.3173
2025-03-23 17:05:49,832 - Class Level Recall: 0.1201
2025-03-23 17:05:49,832 - Class Level F1 Score: 0.1524
2025-03-23 17:05:49,832 - 
Region 15:
2025-03-23 17:05:49,832 - Sample Level Precision: 0.4724
2025-03-23 17:05:49,832 - Sample Level Recall: 0.4042
2025-03-23 17:05:49,832 - Sample Level F1 Score: 0.4185
2025-03-23 17:05:49,832 - Class Level Precision: 0.2980
2025-03-23 17:05:49,832 - Class Level Recall: 0.1767
2025-03-23 17:05:49,832 - Class Level F1 Score: 0.2097
2025-03-23 17:05:49,832 - 
Region 16:
2025-03-23 17:05:49,832 - Sample Level Precision: 0.4752
2025-03-23 17:05:49,832 - Sample Level Recall: 0.4098
2025-03-23 17:05:49,832 - Sample Level F1 Score: 0.4229
2025-03-23 17:05:49,832 - Class Level Precision: 0.2950
2025-03-23 17:05:49,833 - Class Level Recall: 0.1827
2025-03-23 17:05:49,833 - Class Level F1 Score: 0.2137
2025-03-23 17:05:49,833 - 
Region 17:
2025-03-23 17:05:49,833 - Sample Level Precision: 0.4664
2025-03-23 17:05:49,833 - Sample Level Recall: 0.3963
2025-03-23 17:05:49,833 - Sample Level F1 Score: 0.4113
2025-03-23 17:05:49,833 - Class Level Precision: 0.3158
2025-03-23 17:05:49,833 - Class Level Recall: 0.1684
2025-03-23 17:05:49,833 - Class Level F1 Score: 0.1985
2025-03-23 17:05:49,833 - 
Region 18:
2025-03-23 17:05:49,833 - Sample Level Precision: 0.4402
2025-03-23 17:05:49,833 - Sample Level Recall: 0.3667
2025-03-23 17:05:49,833 - Sample Level F1 Score: 0.3832
2025-03-23 17:05:49,833 - Class Level Precision: 0.3176
2025-03-23 17:05:49,833 - Class Level Recall: 0.1386
2025-03-23 17:05:49,833 - Class Level F1 Score: 0.1615
2025-03-23 17:05:49,833 - 
Region 19:
2025-03-23 17:05:49,833 - Sample Level Precision: 0.2927
2025-03-23 17:05:49,833 - Sample Level Recall: 0.2831
2025-03-23 17:05:49,833 - Sample Level F1 Score: 0.2862
2025-03-23 17:05:49,833 - Class Level Precision: 0.2635
2025-03-23 17:05:49,833 - Class Level Recall: 0.0494
2025-03-23 17:05:49,833 - Class Level F1 Score: 0.0514
2025-03-23 17:05:49,833 - 
Region 20:
2025-03-23 17:05:49,833 - Sample Level Precision: 0.4404
2025-03-23 17:05:49,833 - Sample Level Recall: 0.3675
2025-03-23 17:05:49,833 - Sample Level F1 Score: 0.3839
2025-03-23 17:05:49,833 - Class Level Precision: 0.2488
2025-03-23 17:05:49,833 - Class Level Recall: 0.1379
2025-03-23 17:05:49,833 - Class Level F1 Score: 0.1611
2025-03-23 17:05:49,833 - 
Region 21:
2025-03-23 17:05:49,833 - Sample Level Precision: 0.4374
2025-03-23 17:05:49,833 - Sample Level Recall: 0.3736
2025-03-23 17:05:49,833 - Sample Level F1 Score: 0.3880
2025-03-23 17:05:49,833 - Class Level Precision: 0.3165
2025-03-23 17:05:49,833 - Class Level Recall: 0.1461
2025-03-23 17:05:49,833 - Class Level F1 Score: 0.1779
2025-03-23 17:05:49,834 - 
Region 22:
2025-03-23 17:05:49,834 - Sample Level Precision: 0.3612
2025-03-23 17:05:49,834 - Sample Level Recall: 0.3145
2025-03-23 17:05:49,834 - Sample Level F1 Score: 0.3255
2025-03-23 17:05:49,834 - Class Level Precision: 0.2183
2025-03-23 17:05:49,834 - Class Level Recall: 0.0802
2025-03-23 17:05:49,834 - Class Level F1 Score: 0.0863
2025-03-23 17:05:49,834 - 
Region 23:
2025-03-23 17:05:49,834 - Sample Level Precision: 0.3821
2025-03-23 17:05:49,834 - Sample Level Recall: 0.3257
2025-03-23 17:05:49,834 - Sample Level F1 Score: 0.3391
2025-03-23 17:05:49,834 - Class Level Precision: 0.2770
2025-03-23 17:05:49,834 - Class Level Recall: 0.0924
2025-03-23 17:05:49,834 - Class Level F1 Score: 0.1031
2025-03-23 17:05:49,834 - 
Region 24:
2025-03-23 17:05:49,834 - Sample Level Precision: 0.3588
2025-03-23 17:05:49,834 - Sample Level Recall: 0.3145
2025-03-23 17:05:49,834 - Sample Level F1 Score: 0.3247
2025-03-23 17:05:49,834 - Class Level Precision: 0.3450
2025-03-23 17:05:49,834 - Class Level Recall: 0.0796
2025-03-23 17:05:49,834 - Class Level F1 Score: 0.0857
2025-03-23 17:05:49,834 - 
Region 25:
2025-03-23 17:05:49,834 - Sample Level Precision: 0.4381
2025-03-23 17:05:49,834 - Sample Level Recall: 0.3644
2025-03-23 17:05:49,834 - Sample Level F1 Score: 0.3811
2025-03-23 17:05:49,834 - Class Level Precision: 0.2509
2025-03-23 17:05:49,834 - Class Level Recall: 0.1362
2025-03-23 17:05:49,834 - Class Level F1 Score: 0.1603
2025-03-23 17:05:49,834 - 
Region 26:
2025-03-23 17:05:49,834 - Sample Level Precision: 0.4645
2025-03-23 17:05:49,834 - Sample Level Recall: 0.3959
2025-03-23 17:05:49,834 - Sample Level F1 Score: 0.4103
2025-03-23 17:05:49,834 - Class Level Precision: 0.3215
2025-03-23 17:05:49,834 - Class Level Recall: 0.1692
2025-03-23 17:05:49,834 - Class Level F1 Score: 0.1983
2025-03-23 17:05:49,834 - 
Region 27:
2025-03-23 17:05:49,834 - Sample Level Precision: 0.4057
2025-03-23 17:05:49,835 - Sample Level Recall: 0.3445
2025-03-23 17:05:49,835 - Sample Level F1 Score: 0.3588
2025-03-23 17:05:49,835 - Class Level Precision: 0.3262
2025-03-23 17:05:49,835 - Class Level Recall: 0.1076
2025-03-23 17:05:49,835 - Class Level F1 Score: 0.1329
2025-03-23 17:05:49,835 - 
Region 28:
2025-03-23 17:05:49,835 - Sample Level Precision: 0.4701
2025-03-23 17:05:49,835 - Sample Level Recall: 0.4026
2025-03-23 17:05:49,835 - Sample Level F1 Score: 0.4166
2025-03-23 17:05:49,835 - Class Level Precision: 0.3674
2025-03-23 17:05:49,835 - Class Level Recall: 0.1751
2025-03-23 17:05:49,835 - Class Level F1 Score: 0.2082
"""

# Lists for storing data
regions = []
precision = []
recall = []
f1_score = []

# Parse text data
lines = text.split('\n')
current_region = None
for line in lines:
    match_region = re.match(r'Region (\d+):', line)
    if match_region:
        current_region = int(match_region.group(1))
        regions.append(current_region)
    match_precision = re.match(r'.*Sample Level Precision: (\d+\.\d+)', line)
    if match_precision:
        precision.append(float(match_precision.group(1)))
    match_recall = re.match(r'.*Sample Level Recall: (\d+\.\d+)', line)
    if match_recall:
        recall.append(float(match_recall.group(1)))
    match_f1 = re.match(r'.*Sample Level F1 Score: (\d+\.\d+)', line)
    if match_f1:
        f1_score.append(float(match_f1.group(1)))

# Create bar chart
width = 0.2
x = range(len(regions))

plt.bar([i - width for i in x], precision, width=width, label='Sample Level Precision')
plt.bar(x, recall, width=width, label='Sample Level Recall')
plt.bar([i + width for i in x], f1_score, width=width, label='Sample Level F1 Score')

plt.xticks(x, regions)
plt.xlabel('Anatomical Region')
plt.ylabel('Metric Value')
plt.title('Sample Level Metrics by Anatomical Region')
plt.legend()

plt.savefig('test.png')