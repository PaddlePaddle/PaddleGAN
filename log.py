o1_filepath = './log_v6_o2_nan'
#o2_filepath = './ppyolo_o2.detail'
#filepath = './amp_o2_black_2.log'

def get_fp16_op(filepath):
    print(filepath)
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    fp16_op = set()
    op_dict = dict()
    for line in lines:
        op = ''
        if 'API kernel key' in line and 'float16' in line:
            op = line.split(' ')[5]
        if 'Dynamic mode PrepareImpl' in line:
            kernel = line.split('kernel key: ')[1].split(' | kernel: ')[0]
            if 'float16' in kernel:
                op = line.split('kernel name: ')[1].split(' | kernel key')[0]
        if op != '' and '_grad' not in op:
            fp16_op.add(op)
            if op in op_dict.keys():
                op_dict[op] += 1
            else:
                op_dict[op] = 1
    
    print(sorted(fp16_op))
    #print(op_dict)
    return fp16_op


o1_ops = get_fp16_op(o1_filepath)
#o2_ops = get_fp16_op(o2_filepath)
#o2_ops = get_fp16_op(filepath)
print('========= additional ops ============')
#print(sorted(o2_ops.difference(o1_ops)))
