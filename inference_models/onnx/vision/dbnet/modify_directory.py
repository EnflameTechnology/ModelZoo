import re
f = open('instances_test.json', 'r')
str1 = f.read()
str2 = re.sub('test/(gt_img_\d+.txt)', 'ch4_test_localization_transcription_gt/\\1', str1)
str2 = re.sub('test/(img_\d+.jpg)', 'ch4_test_images/\\1', str2)
f2 = open('instances_test.json', 'w')
f2.write(str2)
