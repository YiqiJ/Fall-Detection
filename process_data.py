import os
import math

input_path = 'raw_data'
output_path = 'data'

def to_real(line):
    result = []
    for i in range(len(line) / 2):
        a, b = float(line[i*2]), float(line[i*2+1][:-1])
        real = math.sqrt(a * a + b * b)
        result.append(str(real))
    result = ' '.join(result)
    return result

for folder in os.listdir(input_path):
    cur_input_path = os.path.join(os.getcwd(), input_path, folder)
    cur_output_path = os.path.join(os.getcwd(), output_path, folder)
    if os.path.isdir(cur_input_path):
        for input_file in os.listdir(cur_input_path):
            cur_input_file = os.path.join(cur_input_path, input_file)
            cur_output_file = os.path.join(cur_output_path, input_file)
            with open(cur_input_file) as input_f:
                output_f = open(cur_output_file, 'a+')
                for line in input_f.readlines():
                    line = line.strip()
                    line = line.split(' ')
                    line = to_real(line)
                    output_f.write('{}\n'.format(line))
                output_f.close()
